from __future__ import annotations

import fisea

def test_add():
    assert fisea.functional.add(1, 2) == 3


def test_sub():
    assert fisea.functional.sub(1, 2) == -1