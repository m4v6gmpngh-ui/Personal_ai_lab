import unittest

from personal_ai.core.event import Event
from personal_ai.core.working_state import WorkingState


class WorkingStateTests(unittest.TestCase):
    def test_emotion_carries_for_bounded_following_turns(self) -> None:
        state = WorkingState(emotion_ttl=2)
        state.advance(Event("I am afraid.", emotions={"fear"}))

        first, carried_first = state.enrich(Event("What happened?"))
        self.assertEqual(first.emotions, {"fear"})
        self.assertEqual(carried_first, {"emotion": ["fear"]})
        state.advance(Event("What happened?"))

        second, carried_second = state.enrich(Event("And then?"))
        self.assertEqual(second.emotions, {"fear"})
        self.assertEqual(carried_second, {"emotion": ["fear"]})
        state.advance(Event("And then?"))

        third, carried_third = state.enrich(Event("Anything else?"))
        self.assertEqual(third.emotions, set())
        self.assertEqual(carried_third, {})

    def test_current_emotion_overrides_carried_emotion(self) -> None:
        state = WorkingState(emotion_ttl=3)
        state.advance(Event("I am afraid.", emotions={"fear"}))

        recall_event, carried = state.enrich(
            Event("I feel calm now.", emotions={"calm"})
        )

        self.assertEqual(recall_event.emotions, {"calm"})
        self.assertEqual(carried, {})

        state.advance(Event("I feel calm now.", emotions={"calm"}))
        next_event, next_carried = state.enrich(Event("What now?"))
        self.assertEqual(next_event.emotions, {"calm"})
        self.assertEqual(next_carried, {"emotion": ["calm"]})


if __name__ == "__main__":
    unittest.main()
