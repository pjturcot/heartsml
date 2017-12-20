import pytest

from heartsml.core import Card, Deck, string2card, Trick, Hearts


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


def test_trick_valid_plays_first_trick():
    t = Trick(False, True)  # Hearts not broken, first trick
    all_cards = Deck()
    all_cards.initialize()
    valid_plays = t.valid_plays(all_cards)
    assert len(valid_plays) == 1
    assert Card(2, Card.CLUBS) in valid_plays

    t.append( Card( 2, Card.CLUBS ))    # Leading clubs
    assert t.lead_suit == Card.CLUBS

    valid_plays = t.valid_plays(all_cards)
    assert len(valid_plays) == 13   # All clubs (since clubs are lead)

    # Check void
    cards = map( string2card, ['5H','5D','5S'] )
    valid_cards = t.valid_plays(cards)
    assert set(valid_cards) == set( map(string2card, ['5D','5S']))

def test_trick_valid_plays_latertrick_heartsnotbroken():
    t = Trick( False, False )

    all_cards = Deck()
    all_cards.initialize()
    valid_plays = t.valid_plays(all_cards)
    assert len(valid_plays) == (52 - 13 - 1)  # All cards less point cards
    assert all(not Hearts.is_point_card(c) for c in valid_plays)

    # Check that hearts is valid if it's the only card remaining
    cards = map( string2card, ['5H','QS', 'AH' ])
    valid_plays = t.valid_plays(cards)
    assert set(valid_plays) == set( map(string2card, ['5H','QS', 'AH' ]))

    t.append( Card( 4, Card.DIAMONDS) )
    assert t.lead_suit == Card.DIAMONDS

    valid_plays = t.valid_plays(all_cards)
    assert len(valid_plays) == 13  # All diamonds are playable

    # Check smaller hand
    cards = map( string2card, ['5H','5D','5S'] )
    valid_cards = t.valid_plays(cards)
    assert set(valid_cards) == set( map(string2card, ['5D']))

    # Check void
    cards = map( string2card, ['5H','6C','5S'] )
    valid_plays = t.valid_plays(cards)
    assert set(valid_plays) == set( map(string2card, ['5H', '6C','5S']))

def test_trick_valid_plays_hearts_broken():
    with pytest.raises( RuntimeError ):
        t = Trick( True, True )
    t = Trick( True, False )
    all_cards = Deck()
    all_cards.initialize()
    valid_plays = t.valid_plays(all_cards)
    assert len(valid_plays) == 52   # All cards playable now that hearts are broken


