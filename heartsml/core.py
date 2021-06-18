# coding: utf8

import copy
import logging
import random

import numpy as np

import UCT


class Card(object):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3

    SUIT_NAME = {CLUBS: "Clubs", DIAMONDS: "Diamonds", HEARTS: "Hearts", SPADES: "Spades"}
    SUITS = SUIT_NAME.keys()
    VALUE_NAME = {1: "Ace", 10: "Ten", 11: "Jack", 12: "Queen", 13: "King"}
    VALUE_NAME.update(dict([(i, str(i)) for i in range(2, 10)]))
    VALUES = VALUE_NAME.keys()

    def __init__(self, value=None, suit=None, index=None):
        if index is None:
            assert value in self.VALUE_NAME
            assert suit in self.SUIT_NAME
            self.suit = suit
            self.value = value
            self.index = self.suit * 13 + self.value - 1
        else:
            assert index >= 0 and index < 52
            self.suit = index / 13
            self.value = index % 13 + 1
            self.index = index

    def __repr__(self):
        return self.VALUE_NAME[self.value][0] + self.SUIT_NAME[self.suit][0]

    def __cmp__(self, other):
        ret = self.suit.__cmp__(other.suit)
        if ret == 0:
            ret = self.value.__cmp__(other.value)
        return ret

    def asarray(self, **kwargs):
        a = np.zeros((52,), **kwargs)
        a[self.index] = 1
        return a


CARD_LOOKUP = {}
for suit, suit_name in Card.SUIT_NAME.iteritems():
    for value, value_name in Card.VALUE_NAME.iteritems():
        CARD_LOOKUP[suit_name[0] + value_name[0]] = Card(value, suit)
        CARD_LOOKUP[value_name[0] + suit_name[0]] = Card(value, suit)


def string2card(cardstr):
    if cardstr not in CARD_LOOKUP:
        raise ValueError("Unrecognized card string: {value}".format(value=cardstr))
    return CARD_LOOKUP[cardstr]


class UnorderedDeck():
    """Class to represent a deck of cards as a single binary vector."""

    def __init__(self, cards=None):
        self._array = np.zeros((52,), dtype='bool')
        if cards is not None:
            for c in cards:
                self._array[c.index] = True

    def __repr__(self):
        d = Deck()
        for c in self:
            d.append(c)
        return d.__repr__()

    def __iter__(self):
        """Iterate through cards."""
        for c in np.nonzero(self._array)[0]:
            yield Card(index=c)

    def __contains__(self, card):
        return self._array[card.index]

    def __and__(self, other_deck):
        d = UnorderedDeck()
        d._array = self.asarray() & other_deck.asarray()
        return d

    def __or__(self, other_deck):
        d = UnorderedDeck()
        d._array = self.asarray() | other_deck.asarray()

    def __add__(self, other_deck):
        """Add is treated as set-addition."""
        return self.__or__(other_deck)

    def __sub__(self, other_deck):
        """Subtract is a set subtraction."""
        d = UnorderedDeck()
        d._array = (self.asarray() & ~other_deck.asarray())
        return d

    def __len__(self):
        return np.sum(self._array)

    def append(self, card):
        assert self._array[card.index] == False
        self._array[card.index] = True

    def remove(self, card):
        assert self._array[card.index] == True
        self._array[card.index] = False

    def copy(self):
        d = UnorderedDeck()
        d._array = self._array.copy()
        return d

    def asarray(self):
        return self._array


class Deck(list):
    def __init__(self):
        pass

    def __repr__(self):
        return self.tostring()

    def copy(self):
        return copy.deepcopy(self)

    def initialize(self):
        for suit in Card.SUIT_NAME.keys():
            for value in Card.VALUE_NAME.keys():
                self.append(Card(value, suit))

    def tostring(self, sort=True):
        if sort:
            sorted_cards = sorted(self)
            current_suit = None
            current_suit_cards = []
            output = ""
            for card in sorted_cards:
                if card.suit != current_suit:
                    if current_suit is not None:
                        output += "{suit}: {values} ".format(suit=Card.SUIT_NAME[current_suit],
                                                             values=str.join(",", [Card.VALUE_NAME[x.value][0] for x in current_suit_cards]))
                    current_suit_cards = []
                    current_suit = card.suit
                current_suit_cards.append(card)
            if current_suit is not None:
                output += "{suit}: {values} ".format(suit=Card.SUIT_NAME[current_suit],
                                                     values=str.join(",", [Card.VALUE_NAME[x.value][0] for x in current_suit_cards]))
        else:
            output = str.join(",", map(repr, self))
        return output

    def shuffle(self):
        random.shuffle(self)

    def asarray(self, **kwargs):
        return np.array(map(lambda x: x.asarray(**kwargs), self))


class Player():
    def __init__(self, id=None):
        self.id = id  # Identifier used for the player
        self.points = 0  # Number of points accumulated by the player
        self.hand = UnorderedDeck()
        self.cards_won = UnorderedDeck()

    def __repr__(self):
        return "Player {id} ({points} points): {cards}  (Cards won: {won})".format(
            id=self.id, points=self.points, cards=self.hand, won=self.cards_won)

    def copy(self):
        p = Player()
        p.id = self.id
        p.points = self.points
        p.hand = self.hand.copy()
        p.cards_won = self.cards_won.copy()
        return p


class Trick(Deck):
    def __repr__(self):
        return "Trick: " + self.tostring(sort=False)

    def copy(self):
        t = copy.deepcopy(self)
        return t

    def append(self, card):
        if len(self) == 0:
            self.lead_suit = card.suit
        elif len(self) >= 4:
            raise RuntimeError("Unable to append a 5th card to trick!")
        if card.suit == Card.HEARTS:
            self.hearts_broken = True
        super(Trick, self).append(card)

    def winner(self):
        if len(self) != 4:
            raise RuntimeError("Cannot determine a winner until 4 cards are played.")
        return np.argmax([(card.value if card.value != 1 else 14) if card.suit == self.lead_suit else -1 for card in self])

    def asarray(self):
        a = []
        for c in self:
            a.append(c.asarray())
        while len(a) < 4:
            a.append( np.zeros((52,)))
        return np.array(a)



class HeartsState():
    POINT_CARD_MASK = UnorderedDeck()
    for value in Card.VALUES:
        POINT_CARD_MASK.append(Card(suit=Card.HEARTS, value=value))
    POINT_CARD_MASK.append(Card(suit=Card.SPADES, value=12))

    POINT_VALUES_MASK = POINT_CARD_MASK.asarray().astype('float')
    POINT_VALUES_MASK[Card(suit=Card.SPADES, value=12).index] += 12

    SUIT_MASK = dict()
    for suit in Card.SUITS:
        d = UnorderedDeck()
        for value in Card.VALUES:
            d.append(Card(suit=suit, value=value))
        SUIT_MASK[suit] = d

    TWO_CLUBS_MASK = UnorderedDeck()
    TWO_CLUBS_MASK.append(Card(suit=Card.CLUBS, value=2))

    def __init__(self, max_score=50, randomstate=None):
        if randomstate is None:
            self.randomstate = np.random.RandomState(None)
        else:
            self.randomstate = randomstate
        self.max_score=max_score
        self.reset()

    def count_points(self, unordered_deck):
        return (unordered_deck.asarray() * self.POINT_VALUES_MASK).sum()

    def _deal_cards(self):
        """Deal a round of cards to all the players."""
        for p in self.players:
            assert len(p.hand) == 0
            p.cards_won = UnorderedDeck()

        d = Deck()
        d.initialize()
        self.randomstate.shuffle( d )
        for i_round in range(13):
            for ip, p in enumerate(self.players):
                p.hand.append(d.pop())
        self.turn = -1
        self.trick = Trick()

        self.leading_player = (self._player_index(Card(2, Card.CLUBS))) % 4
        self.playerJustMoved = self.leading_player

        assert len(d) == 0

    def _player_index(self, card):
        """Find which player is holding a specific card."""
        for ip, p in enumerate(self.players):
            if card in p.hand:
                return ip

    def hearts_broken(self):
        """Return: whether hearts are broken or not."""
        return (self.all_cards_played() & self.POINT_CARD_MASK.asarray()).any()

    def all_cards_played(self):
        """Return a vector of all cards played so far."""
        return np.array([p.cards_won.asarray() for p in self.players]).any(axis=0)

    def valid_plays(self, unordered_deck):

        playable_cards = UnorderedDeck()
        if self.round() == 0 and len(self.trick) == 0:  # First play
            return self.TWO_CLUBS_MASK & unordered_deck
        elif len(self.trick):  # If a card has been played already.. try and follow suit
            playable_cards = self.SUIT_MASK[self.trick[0].suit] & unordered_deck

            if len(playable_cards) == 0:
                # Can't follow suit
                playable_cards = unordered_deck.copy()
                if self.round() == 0:
                    playable_cards -= self.POINT_CARD_MASK
        else:  # Leading card
            assert self.round() > 0 and len(self.trick) == 0
            playable_cards = unordered_deck.copy()
            if not self.hearts_broken():
                playable_cards -= self.POINT_CARD_MASK

        if len(playable_cards) == 0:
            playable_cards = unordered_deck.copy()  # Play hearts
            assert len(playable_cards & self.POINT_CARD_MASK) == len(playable_cards)  # Check that all cards are point cards at this point

        return list(playable_cards)

    def round(self):
        """What is the current round we are playing."""
        return self.all_cards_played().sum() / 4

    def reset(self):
        self.players = [Player(id) for id in range(4)]
        self._deal_cards()

    def Clone(self):
        """Create a deep copy of the HeartsState"""
        st = HeartsState( randomstate=copy.deepcopy( self.randomstate ))
        st.players = [p.copy() for p in self.players]
        st.turn = self.turn
        st.trick = self.trick.copy()
        st.leading_player = self.leading_player
        st.playerJustMoved = self.playerJustMoved
        st.max_score = self.max_score
        return st

    def DoMove(self, card):
        possible_plays = self.GetMoves()
        self.turn += 1
        self.playerJustMoved = (self.leading_player + self.turn) % 4
        logging.debug("[Round {round}] {trick: <18}. P{player} possible plays: {plays: <50} .... Chose: {card}".format(
            trick=self.trick, player=self.playerJustMoved, plays=possible_plays, card=card, round=self.round()))
        self.players[self.playerJustMoved].hand.remove(card)
        self.trick.append(card)

        if len(self.trick) == 4:
            winning_player = (self.trick.winner() + self.leading_player) % 4
            logging.info("[Round {round}] {trick: <18}  WINNER: {winner} (P{player})".format(
                round=self.round(), trick=self.trick, player=winning_player, winner=self.trick[self.trick.winner()]))
            self.leading_player = winning_player
            for c in self.trick:
                self.players[winning_player].cards_won.append(c)
            self.turn = -1
            self.trick = Trick()

            if len(self.players[self.playerJustMoved].hand) == 0:   # No more cards... game over?
                points = np.array([self.count_points(p.cards_won) for p in self.players])
                if any(points == 26):
                    points = -points + 26
                for p, player in zip( points.tolist(), self.players ):
                    player.points += p
                if not self.is_game_over():
                    logging.info("Points by player so far in the game: {points}".format(
                        points = [ p.points for p in self.players ] ))
                    logging.info("Dealing another round of cards.")
                    self._deal_cards()

    def is_game_over(self):
        player_points = np.array( [ p.points for p in self.players] )
        return (player_points >= self.max_score).any()

    def current_player(self):
        return (self.leading_player + self.turn + 1) % 4

    def GetMoves(self):
        possible_plays = self.valid_plays(self.players[self.current_player()].hand)
        return list(possible_plays)

    def GetResult(self, player_index, result_type='winner'):
        player_points = np.array( [ p.points for p in self.players] )
        if result_type == 'winner':
            winners = player_points == player_points.min()
            n_winners = winners.sum()
            win_value = np.ones( player_points.shape )* -1.0 / ( 4 - n_winners )
            win_value[ winners ] = 1.0 / n_winners
            return win_value[player_index]
        elif result_type == 'winloss':
            winners = player_points == player_points.min()
            losers = player_points == player_points.max()
            win_value = np.zeros( player_points.shape )
            win_value[ winners ] = 1.0 / winners.sum()
            win_value[ losers ] = -1.0 / losers.sum()
            return win_value[player_index]
        elif result_type == 'all_winloss':
            winners = player_points == player_points.min()
            losers = player_points == player_points.max()
            win_value = np.zeros( player_points.shape )
            win_value[ winners ] = 1.0 / winners.sum()
            win_value[ losers ] = -1.0 / losers.sum()
            player_order = (np.arange(4) + player_index) % 4
            return win_value[ player_order ]
        elif result_type == 'all_rawpoints':
            player_order = (np.arange(4) + player_index) % 4
            result = np.ones( player_points.shape ) * player_points.mean() - player_points  # Center the player points so they are 0
            return result[ player_order ]
        elif result_type == 'rawpoints':
            result = np.ones( player_points.shape ) * player_points.mean() - player_points  # Center the player points so they are 0
            return result[player_index]
        else:
            raise ValueError("Unknown result type: {result_type}".format(result_type=result_type) )


    def __repr__(self):
        return "[Round {round}].\n{trick}\n{player}".format(
            round=self.round(), trick=self.trick, player=self.players[self.current_player()])

    def display(self, display_players=set([0])):
        kwargs = {}
        for i in range(4):
            player_id = (self.leading_player + i) % 4
            if i < len( self.trick ):
                kwargs['player_{id}'.format( id=player_id) ] = self.trick[i]
            else:
                kwargs['player_{id}'.format( id=player_id) ] = "??" if player_id == self.current_player() else "__"
        trick_lines = """
 |        [ {player_2:2} ]       |
 |                     |
 |   [ {player_1:2} ]   [ {player_3:2} ]   |
 |                     |
 |        [ {player_0:2} ]       |""".format( **kwargs ).split('\n')

        PLAYER_TEMPLATE = """
  Player {id}
  Hand:
  {hand}
  Cards Won:
  {cards_won}
"""
        player_lines = {}
        for player_id, player in enumerate(self.players):
            if player_id in display_players:
                hand = str( player.hand )
                cards_won = str( player.cards_won )
            else:
                hand = "?? ({n} cards)".format(n=len(player.hand))
                cards_won = "?? ({n} cards)".format(n=len(player.cards_won))
            player_lines[player_id] = PLAYER_TEMPLATE.format( id=player_id, hand=hand, cards_won=cards_won ).split('\n')

        player_1_width = max( map(len, player_lines[1]))
        trick_width = max( map(len, trick_lines))

        # Display Player 2
        for l in player_lines[2]:
            print " "*player_1_width + l
        # Display Player 1 , trick and Player 3
        for p1, trick, p3 in zip( player_lines[1], trick_lines, player_lines[3]):
            print p1 + " "*(player_1_width - len(p1)) + trick + " "*(trick_width-len(trick)) + p3
        for l in player_lines[0]:
            print " "*player_1_width + l
