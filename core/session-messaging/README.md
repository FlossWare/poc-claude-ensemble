# Claude Ensemble Session Messaging

claude-messenger is a small local pub/sub service for session-to-session commands. It is deliberately separate from the memory service: memory stores state, while this service delivers transient events.

## Protocol

Clients send newline-delimited JSON over a Unix domain socket.

Legacy topic subscription:

    {"op":"subscribe","topic":"credentials"}

Explicit event subscription:

    {
      "op":"subscribe",
      "subscription_id":"credential-reloads",
      "filter":{"event_types":["credentials.reload"]}
    }

Topic and event type can be combined. Both conditions must match:

    {
      "op":"subscribe",
      "subscription_id":"credential-reloads",
      "filter":{
        "topics":["credentials"],
        "event_types":["credentials.reload"]
      }
    }

Publish:

    {
      "op":"publish",
      "topic":"credentials",
      "event_type":"credentials.reload",
      "data":{"action":"reload"}
    }

If `event_type` is omitted, it defaults to `topic` for backward compatibility.

### Subscription contract

A subscription filter is a non-empty object containing one or both of:

- `topics`: exact topic names
- `event_types`: exact event type names

Values are case-sensitive strings. Empty or unknown filter fields are rejected. Filtering is performed by the messenger before delivery, so an unrelated subscriber does not receive the event and cannot accidentally process it.

A client may maintain multiple subscriptions over one connection. `subscription_id` identifies each subscription and is required when unsubscribing. If multiple subscriptions on the same connection match one event, the event is delivered once to that connection, not once per matching subscription.

Published events are delivered to matching current subscribers only and are not persisted.

## Security and lifecycle

- The socket defaults to `$XDG_RUNTIME_DIR/claude-messenger/claude-messenger.sock`.
- The service creates the socket with mode 0600.
- The systemd template places it under the user runtime directory with 0700 ownership.
- No TCP listener or external dependency is used.
- Disconnected subscribers are removed.
- Clients can reconnect automatically after a service restart.

## Running

    cd session-messaging
    python3 -m unittest -v

Install the user service (the template uses @ENSEMBLE_ROOT@ as an install-time placeholder):

    mkdir -p ~/.config/systemd/user
    sed "s|@ENSEMBLE_ROOT@|$HOME/Development/FlossWare/claude-ensemble|g" claude-messenger.service.template > ~/.config/systemd/user/claude-messenger.service
    systemctl --user daemon-reload
    systemctl --user enable --now claude-messenger.service

Publish:

    python3 -c 'from messenger_client import MessengerClient; MessengerClient().publish("credentials", {"action": "reload"})'

Listen for a specific event type:

    python3 -c 'from messenger_client import MessengerClient; print(next(MessengerClient().subscribe_events(["credentials.reload"])))'

The client module is intentionally dependency-free so it can be used by shell/session initialization without another service framework.
