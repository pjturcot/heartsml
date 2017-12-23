import pytest

from heartsml.core import Card, Deck, UnorderedDeck, string2card, Trick, HeartsState


def test_string2card():
    assert string2card('AH') == Card(1, Card.HEARTS)
    assert string2card('HA') == Card(1, Card.HEARTS)
    assert string2card('KD') == Card(13, Card.DIAMONDS)
    assert string2card('TC') == Card(10, Card.CLUBS)

    with pytest.raises(ValueError):
        c = string2card('2X')


def test_deck():
    d = Deck()
    assert len(d) == 0
    d.initialize()
    assert len(d) == 52

    d2 = Deck()
    d2.append(d.pop())

    assert len(d) == 51
    assert len(d2) == 1

def test_unordered_deck():
    d = UnorderedDeck()
    assert len(d) == 0
    d.append( Card( suit=Card.HEARTS, value=2 ))
    assert len(d) == 1

    assert Card(suit=Card.HEARTS, value=2) in d
    s = d.__repr__()

    d.append( Card( suit=Card.HEARTS, value=1))
    cards = list( d )
    assert len(d) == 2
    assert len(cards) == 2

    d._array[:] = True
    s = d.__repr__()