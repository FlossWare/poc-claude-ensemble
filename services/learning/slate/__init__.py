"""Slate (crew) selection bandits.

Crew selection currently draws worker arms independently.  A review,
however, succeeds or fails as a CREW: three redundant workers from one
model family waste a slot that a diverse crew would have used.  This
package scores whole slates: per-arm Beta posteriors supply marginal
quality, and a diversity bonus rewards slates that span model families
(a linear-submodular relaxation, greedy near-optimal).
"""
