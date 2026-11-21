#!/usr/bin/env python3
"""
Claude Ensemble Thompson Router Service Daemon

Central Thompson Sampling authority for model selection across Claude Code sessions.
Runs as a loopback HTTP/JSON service and owns its persisted sampling state.
"""

import json
import logging
import socket
import sys
import os
import tempfile
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional, Any
import uuid
import numpy as np
from scipy.stats import beta as beta_dist

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.request_context import RequestContext
from shared.validators import Validators
from shared.runtime_config import log_dir, socket_path, thompson_state_file
from shared.service_http import start_service_http
from shared.thompson_policy import (
    Arm,
    default_params,
    load_policy_file,
    policy_for_task,
    select as policy_select,
)

LOG_DIR = log_dir()
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / 'claude-thompson.log')
    ]
)
logger = logging.getLogger(__name__)

SOCKET_PATH = socket_path('POC_CLAUDE_ENSEMBLE_THOMPSON_SOCKET', '/tmp/claude-thompson.sock')
HTTP_PORT = int(os.environ.get('POC_CLAUDE_ENSEMBLE_THOMPSON_HTTP_PORT', '8769'))
STATE_FILE = thompson_state_file()


class ModelStats:
    """Statistics for a single model"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.successes = 0
        self.failures = 0
        # Discounted (effective) counts: raw history is preserved above for
        # reporting; the router draws from these so old evidence decays when
        # POC_CLAUDE_ENSEMBLE_THOMPSON_DISCOUNT < 1.0 (1.0 = classic Thompson).
        self.eff_successes = 0.0
        self.eff_failures = 0.0
        self.total_cost = 0.0
        self.total_tokens = 0
        self.calls = 0
        self.last_updated = datetime.now(timezone.utc).isoformat()
        # External prior provenance when the arm was seeded (LMSYS ELO and
        # similar); empty for arms built from real outcomes only.
        self.prior: Dict[str, Any] = {}

    @property
    def success_rate(self) -> float:
        """Success rate for Beta distribution prior"""
        if self.calls == 0:
            return 0.5  # Neutral prior
        return self.successes / self.calls

    @property
    def avg_cost(self) -> float:
        """Average cost per call"""
        if self.calls == 0:
            return 0.0
        return self.total_cost / self.calls

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict"""
        return {
            'model_name': self.model_name,
            'successes': self.successes,
            'failures': self.failures,
            'eff_successes': self.eff_successes,
            'eff_failures': self.eff_failures,
            'total_cost': self.total_cost,
            'total_tokens': self.total_tokens,
            'calls': self.calls,
            'last_updated': self.last_updated,
            'prior': self.prior,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelStats':
        """Deserialize from dict"""
        model = cls(data['model_name'])
        model.successes = data.get('successes', 0)
        model.failures = data.get('failures', 0)
        # Older state files predate the effective counts: seed them from the
        # raw counts so the first discounted update starts from full history.
        model.eff_successes = float(data.get('eff_successes', model.successes))
        model.eff_failures = float(data.get('eff_failures', model.failures))
        model.total_cost = data.get('total_cost', 0.0)
        model.total_tokens = data.get('total_tokens', 0)
        model.calls = data.get('calls', 0)
        model.last_updated = data.get('last_updated', datetime.now(timezone.utc).isoformat())
        prior = data.get('prior')
        model.prior = prior if isinstance(prior, dict) else {}
        return model


class ThompsonState:
    """Manages Thompson Sampling state with file persistence"""

    def __init__(self, state_file: Path):
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, ModelStats] = {}
        self.last_updated = datetime.now(timezone.utc).isoformat()
        self._load()

    def _load(self):
        """Load state from file.

        A corrupt file is moved aside (``.corrupt-<timestamp>``) before the
        service starts fresh, so the next save can never overwrite the only
        copy of the learned history.  Individual malformed model records are
        skipped -- one bad record must not discard every model.
        """
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)

                self.last_updated = data.get('last_updated', datetime.now(timezone.utc).isoformat())

                # Load model stats, skipping (not failing on) bad records
                models = data.get('models', {})
                if not isinstance(models, dict):
                    raise ValueError(f"'models' is not an object: {type(models).__name__}")
                skipped = 0
                for model_data in models.values():
                    try:
                        stats = ModelStats.from_dict(model_data)
                        self.models[stats.model_name] = stats
                    except (KeyError, TypeError, ValueError) as record_exc:
                        skipped += 1
                        logger.warning(
                            f"Skipping malformed model record ({record_exc})")
                if skipped:
                    logger.warning(f"{skipped} model record(s) skipped during load")

                logger.info(f"Loaded state for {len(self.models)} models")
            except Exception as e:
                logger.error(f"Error loading state: {e}")
                # Preserve the corrupt file for diagnosis/recovery before
                # starting fresh; the next save would otherwise destroy it.
                backup = self.state_file.with_name(
                    f"{self.state_file.name}.corrupt-{int(time.time())}")
                try:
                    os.replace(self.state_file, backup)
                    logger.warning(f"Corrupt state file moved aside: {backup}")
                except OSError as move_exc:
                    logger.error(f"Could not move corrupt state aside: {move_exc}")
                self.models = {}
        else:
            logger.info(f"State file not found, starting fresh: {self.state_file}")

    def save(self) -> bool:
        """Save state to file with atomic write.

        Returns True on success.  A failed save is returned as False (and
        logged) so callers like ``record_outcome`` can report the outcome as
        NOT persisted instead of acknowledging lost data.
        """
        try:
            self.last_updated = datetime.now(timezone.utc).isoformat()

            data = {
                'last_updated': self.last_updated,
                'models': {
                    model_name: stats.to_dict()
                    for model_name, stats in self.models.items()
                }
            }

            # Atomic write: temp file + rename.  The temp file is unlinked
            # on failure so a full disk does not litter the state directory.
            tmp_name = None
            try:
                with tempfile.NamedTemporaryFile(mode='w', dir=self.state_file.parent, delete=False) as tmp:
                    tmp_name = tmp.name
                    json.dump(data, tmp, indent=2)
                    tmp.flush()
                    os.fsync(tmp.fileno())  # Force to disk
                    os.replace(tmp.name, self.state_file)  # Atomic rename
                    tmp_name = None  # replace succeeded; nothing to clean up
            finally:
                if tmp_name is not None:
                    try:
                        os.unlink(tmp_name)
                    except OSError:
                        pass

            logger.info(f"Saved state for {len(self.models)} models")
            return True
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            return False

    def get_or_create_model(self, model_name: str) -> ModelStats:
        """Get or create stats for a model"""
        if model_name not in self.models:
            self.models[model_name] = ModelStats(model_name)
        return self.models[model_name]

    def ensure_arms(self, model_names: list[str]) -> list[str]:
        """Create zero-call arms without recording a fabricated outcome.

        Registration is not evidence: an arm enters with Beta(1, 1) and is
        explored by the normal Thompson draw. Outcomes arrive later from real
        calls. Returns the names that were newly created.
        """
        created: list[str] = []
        for model_name in model_names:
            if model_name not in self.models:
                self.models[model_name] = ModelStats(model_name)
                created.append(model_name)
        if created:
            if not self.save():
                # Roll back so a retry of the failed request cannot
                # double-register; None (not []) signals the failure so the
                # handler can answer ok:false instead of "nothing new".
                for model_name in created:
                    del self.models[model_name]
                logger.error("New arms NOT persisted; registration rolled back")
                return None
            logger.info(f"Registered {len(created)} new arm(s): {created}")
        return created

    def select_model(self, task_type: str, required_capability: float = 0.5, max_cost: float = float('inf')) -> str:
        """
        Select best model using Thompson Sampling.

        The selection algorithm lives in :mod:`shared.thompson_policy` so the
        offline GA replay evaluates exactly the function production runs.  The
        live signature and the HTTP contract are unchanged.

        Args:
            task_type: Type of task (selects the policy entry)
            required_capability: Minimum capability required (0-1)
            max_cost: Maximum cost threshold per call

        Returns:
            Selected model name
        """
        if not self.models:
            logger.warning("No models loaded, returning fallback 'haiku'")
            return 'haiku'

        arms = [
            Arm(name, stats.eff_successes, stats.eff_failures, stats.avg_cost, stats.calls)
            for name, stats in self.models.items()
        ]

        try:
            policies = load_policy_file()
            params = policy_for_task(policies, task_type)
            selected = policy_select(params, arms, task_type, required_capability, max_cost, np.random)
        except Exception as exc:  # noqa: BLE001 - selection must never fail on policy input
            logger.warning(
                f"Ignoring unusable Thompson policy ({type(exc).__name__}: {exc}); "
                "falling back to default parameters"
            )
            selected = policy_select(
                default_params(), arms, task_type, required_capability, max_cost, np.random
            )

        logger.info(f"Selected model: {selected} (task={task_type})")
        return selected

    _DISCOUNT_ENV = "POC_CLAUDE_ENSEMBLE_THOMPSON_DISCOUNT"

    @staticmethod
    def _discount() -> float:
        """Posterior decay factor per outcome, from the environment.

        1.0 (default) = classic cumulative Thompson.  Values in (0, 1) decay
        old evidence exponentially; e.g. 0.99 gives recent outcomes ~63% of
        the posterior after 100 updates.  Invalid values fail closed to 1.0
        (never to something more aggressive than the operator asked for).
        """
        raw = os.environ.get(ThompsonState._DISCOUNT_ENV, "1.0").strip()
        try:
            gamma = float(raw)
        except ValueError:
            logger.warning(f"invalid {ThompsonState._DISCOUNT_ENV} {raw!r}; using 1.0")
            return 1.0
        if not 0.0 < gamma <= 1.0:
            logger.warning(
                f"{ThompsonState._DISCOUNT_ENV} must be in (0, 1]; got {gamma}; using 1.0")
            return 1.0
        return gamma

    def record_outcome(self, model_name: str, task_type: str, success: bool, cost: float, tokens: int) -> bool:
        """
        Record outcome of a model call.

        Args:
            model_name: Model that was used
            task_type: Type of task
            success: Whether the task succeeded
            cost: Cost of the call
            tokens: Tokens used

        Returns:
            True if recorded successfully
        """
        try:
            existed = model_name in self.models
            stats = self.get_or_create_model(model_name)

            # Snapshot for rollback: a failed save must leave in-memory state
            # unmutated, or a client retry of the `ok: false` response would
            # double-count the outcome when a later save succeeds.
            snapshot = stats.to_dict()

            if success:
                stats.successes += 1
            else:
                stats.failures += 1

            # Discounted update: eff = gamma * eff + outcome.  gamma = 1.0
            # (default) reduces exactly to the classic cumulative update, so
            # the stored effective counts equal the raw counts until the
            # operator opts into decay.  gamma in (0, 1) makes the posterior
            # an exponentially-weighted estimate of RECENT performance, so a
            # model that regresses loses rank without a manual reset.
            gamma = self._discount()
            stats.eff_successes = gamma * stats.eff_successes + (1.0 if success else 0.0)
            stats.eff_failures = gamma * stats.eff_failures + (0.0 if success else 1.0)

            stats.total_cost += cost
            stats.total_tokens += tokens
            stats.calls += 1
            stats.last_updated = datetime.now(timezone.utc).isoformat()

            if not self.save():
                if existed:
                    self.models[model_name] = ModelStats.from_dict(snapshot)
                else:
                    del self.models[model_name]  # arm creation rolled back too
                logger.error(f"Outcome for {model_name} NOT persisted; state rolled back")
                return False
            logger.info(f"Recorded outcome for {model_name}: success={success}, cost={cost:.4f}")
            return True
        except Exception as e:
            logger.error(f"Error recording outcome: {e}")
            return False

    def seed_prior(
        self,
        model_name: str,
        alpha: float,
        beta: float,
        *,
        prior_strength: float = 1.0,
        source: Optional[str] = None,
        detail: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Seed an untested arm from an external prior (e.g. LMSYS ELO).

        The stored counts are ``alpha - prior_strength`` and
        ``beta - prior_strength`` because selection adds ``prior_strength``
        (the Beta(1,1)-equivalent) to each count when drawing: the effective
        posterior after seeding is exactly Beta(alpha, beta).

        Refuses (returns False, no mutation) when the arm already has real
        outcomes -- external evidence must never overwrite local evidence --
        or when the arguments cannot represent a valid prior.
        """
        if not isinstance(model_name, str) or not model_name.strip():
            return False
        if isinstance(alpha, bool) or isinstance(beta, bool):
            return False
        if not isinstance(alpha, (int, float)) or not isinstance(beta, (int, float)):
            return False
        if not (alpha > 0 and beta > 0 and prior_strength > 0):
            return False
        successes = float(alpha) - float(prior_strength)
        failures = float(beta) - float(prior_strength)
        if successes < 0 or failures < 0:
            return False
        existing = self.models.get(model_name)
        if existing is not None and existing.calls > 0:
            logger.info(
                f"seed_prior refused for {model_name}: "
                f"{existing.calls} real outcome(s) already recorded")
            return False

        stats = self.get_or_create_model(model_name)
        stats.successes = successes
        stats.failures = failures
        stats.calls = 0
        stats.prior = {
            'alpha': float(alpha),
            'beta': float(beta),
            'prior_strength': float(prior_strength),
            'source': source or 'unspecified',
            'detail': detail or {},
            'seeded_at': datetime.now(timezone.utc).isoformat(),
        }
        stats.last_updated = datetime.now(timezone.utc).isoformat()
        if not self.save():
            # Roll back so the refusal/retry semantics stay clean.
            if existing is None:
                del self.models[model_name]
            else:
                self.models[model_name] = existing
            logger.error(f"seed_prior for {model_name} NOT persisted; rolled back")
            return False
        logger.info(f"Seeded prior for {model_name}: Beta({alpha}, {beta}) from {source}")
        return True

    def reset(self, model_name: str) -> bool:
        """Reset history for a model"""
        try:
            if model_name in self.models:
                snapshot = self.models[model_name].to_dict()
                self.models[model_name] = ModelStats(model_name)
                if not self.save():
                    # Undo: the reset did not persist, so memory must not
                    # claim it happened.
                    self.models[model_name] = ModelStats.from_dict(snapshot)
                    logger.error(f"Reset of {model_name} NOT persisted; rolled back")
                    return False
                logger.info(f"Reset model: {model_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error resetting model: {e}")
            return False

    def get_state(self) -> Dict[str, Any]:
        """Get full state snapshot"""
        return {
            'last_updated': self.last_updated,
            'models': {
                name: stats.to_dict()
                for name, stats in self.models.items()
            }
        }


class ThompsonService:
    """Thompson Router Service Daemon"""

    def __init__(self, socket_path: Path, state_file: Path):
        self.socket_path = Path(socket_path)
        self.state = ThompsonState(state_file)
        self.socket = None

    def start(self):
        """Run the Thompson service over loopback HTTP/JSON only."""
        logger.info("Starting RH Thompson Router Service")
        self.http_server = start_service_http(
            "thompson",
            HTTP_PORT,
            lambda raw: self._process_request(
                raw, RequestContext(caller="thompson-rest", method="http")
            ),
            "action",
        )
        try:
            # The HTTP adapter owns its serving thread. Keep this process alive
            # until shutdown; do not create or accept Unix-domain socket clients.
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            logger.info("Shutting down")
        finally:
            self.stop()

    def stop(self):
        """Stop the REST service cleanly."""
        server = getattr(self, "http_server", None)
        if server is not None:
            server.shutdown()
            server.server_close()
        logger.info("Thompson service stopped")

    def _process_request(self, request: str, ctx: RequestContext) -> str:
        """Process a request, return JSON response"""
        try:
            req_data = json.loads(request)
            action = req_data.get('action')

            # Validate request fields
            is_valid, error_msg = Validators.validate_thompson_request(req_data, action)
            if not is_valid:
                logger.warning(f"{ctx} Validation error: {error_msg}")
                ctx.log_error('validation', Exception(error_msg))
                return json.dumps({'ok': False, 'error': error_msg, 'request_id': ctx.request_id})

            if action == 'select_model':
                task_type = req_data.get('task_type', 'unknown')
                required_capability = req_data.get('required_capability', 0.5)
                max_cost = req_data.get('max_cost', float('inf'))

                model = self.state.select_model(task_type, required_capability, max_cost)
                logger.info(f"{ctx} Model selected: {model}")
                return json.dumps({'ok': True, 'model': model, 'request_id': ctx.request_id})

            elif action == 'record_outcome':
                model = req_data.get('model')
                task_type = req_data.get('task_type', 'unknown')
                success = req_data.get('success', False)
                cost = req_data.get('cost', 0.0)
                tokens = req_data.get('tokens', 0)

                logger.info(f"{ctx} Recording outcome: {model} (success={success}, cost=${cost:.4f})")
                ok = self.state.record_outcome(model, task_type, success, cost, tokens)
                return json.dumps({'ok': ok, 'request_id': ctx.request_id})

            elif action == 'get_state':
                state = self.state.get_state()
                logger.info(f"{ctx} Returning state snapshot")
                return json.dumps({'ok': True, 'state': state, 'request_id': ctx.request_id})

            elif action == 'reset':
                model = req_data.get('model')
                logger.info(f"{ctx} Resetting model: {model}")
                ok = self.state.reset(model)
                return json.dumps({'ok': ok, 'request_id': ctx.request_id})

            elif action == 'ensure_arms':
                models = req_data.get('models') or []
                created = self.state.ensure_arms(models)
                if created is None:
                    logger.error(f"{ctx} Arm registration NOT persisted")
                    return json.dumps({
                        'ok': False,
                        'error': 'ensure_arms failed: state not persisted',
                        'models': models, 'request_id': ctx.request_id})
                logger.info(f"{ctx} Registered arms (new: {len(created)})")
                return json.dumps({'ok': True, 'created': created,
                                   'models': models, 'request_id': ctx.request_id})

            elif action == 'seed_prior':
                model = req_data.get('model')
                alpha = req_data.get('alpha')
                beta = req_data.get('beta')
                prior_strength = req_data.get('prior_strength', 1.0)
                source = req_data.get('source')
                detail = req_data.get('detail')
                logger.info(f"{ctx} Seeding prior for {model} from {source}")
                seeded = self.state.seed_prior(
                    model, alpha, beta,
                    prior_strength=prior_strength, source=source, detail=detail)
                if not seeded:
                    return json.dumps({
                        'ok': False,
                        'error': 'seed_prior refused: invalid arguments or '
                                 'existing evidence on the arm',
                        'request_id': ctx.request_id})
                return json.dumps({
                    'ok': True, 'seeded': True, 'request_id': ctx.request_id})

            elif action == 'ping':
                return json.dumps({'ok': True, 'message': 'pong', 'request_id': ctx.request_id})

            else:
                ctx.log_error('process_request', Exception(f'Unknown action: {action}'))
                return json.dumps({'ok': False, 'error': f'Unknown action: {action}', 'request_id': ctx.request_id})

        except json.JSONDecodeError as e:
            ctx.log_error('json_decode', e)
            return json.dumps({'ok': False, 'error': 'Invalid JSON', 'request_id': ctx.request_id})
        except Exception as e:
            ctx.log_error('process_request', e)
            return json.dumps({'ok': False, 'error': str(e), 'request_id': ctx.request_id})


if __name__ == '__main__':
    service = ThompsonService(SOCKET_PATH, STATE_FILE)
    service.start()
