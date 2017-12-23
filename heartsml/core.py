# coding: utf8

import random
import copy
import logging
import numpy as np
import UCT

class Card(object):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3

    SUIT_NAME = {CLUBS: "Clubs", DIAMONDS: "Diamonds", HEARTS: "Hearts", SPADES: "Spades"}
    VALUE_NAME = {1: "Ace", 10: "Ten", 11: "Jack", 12: "Queen", 13: "King"}
    VALUE_NAME.update(dict([(i, str(i)) for i in range(2, 10)]))

    def __init__(self, value, suit):
        assert value in self.VALUE_NAME
        assert suit in self.SUIT_NAME
        self.suit = suit
        self.value = value
        self.index = self.suit * 13 + self.value - 1

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


class Player(Deck):
    def __init__(self, id):
        self.id = id  # Identifier used for the player
        self.points = 0  # Number of points accumulated by the player

    def __repr__(self):
        return "Player {id}: {cards}".format( id=self.id, cards=self.tostring(sort=True))

    def copy(self):
        p = copy.deepcopy(self)
        p.id = self.id
        p.points = self.points
        return p

class Trick(Deck):

    def __init__(self, hearts_broken, first_trick):
        """Initialize a trick class used to validate what cards are playable.

        :param first_card:   First card in the trick
        :param player_id:
        :param hearts_broken:
        :param first_trick:
        :return:
        """
        self.hearts_broken = hearts_broken
        self.first_trick = first_trick
        self.lead_suit = None
        if self.hearts_broken and self.first_trick:
            raise RuntimeError("You want to explain how a heart has been broken on the first trick?")

    def __repr__(self):
        return "Trick: " + self.tostring(sort=False)

    def copy(self):
        t = copy.deepcopy(self)
        t.hearts_broken = self.hearts_broken
        t.first_trick = self.first_trick
        t.lead_suit = self.lead_suit
        return t

    def append(self, card):
        if len(self) == 0:
            self.lead_suit = card.suit
        elif len(self) >= 4:
            raise RuntimeError("Unable to append a 5th card to trick!")
        if card.suit == Card.HEARTS:
            self.hearts_broken = True
        super(Trick, self).append(card)

    def valid_plays(self, deck):
        """Determine possible valid cards to play.

        :param player:
        :return:
        """
        # Handle the first trick
        if len(self) == 0 and self.first_trick:
            c = Card( 2, Card.CLUBS )
            d = Deck()
            if c in deck:
                d.append(c)
            return d

        point_cards = Deck()
        matching_suit = Deck()
        other_suits = Deck()
        for card in deck:
            if self.lead_suit is not None and card.suit == self.lead_suit:
                matching_suit.append(card)
            elif Hearts.is_point_card(card):
                point_cards.append(card)
            else:
                other_suits.append(card)

        playable_cards = matching_suit
        # Void in matching suit (or leading card)
        if not playable_cards:
            playable_cards += other_suits
            # Add in hearts if (1) they've been broken (2) we're void and it's not the first card
            if (len(self)==0 and self.hearts_broken) or (len(self) > 0 and not self.first_trick):
                playable_cards += point_cards

        # Add in hearts if it's our only option
        if not len(playable_cards):
            playable_cards += point_cards

        return playable_cards

    def winner(self):
        if len(self) != 4:
            raise RuntimeError("Cannot determine a winner until 4 cards are played.")
        return np.argmax([ card.value if card.value != 1 else 14 if card.suit == self.lead_suit else -1 for card in self ])

class Hearts():
    def __init__(self):
        self.reset()

    @staticmethod
    def is_point_card(card):
        return card.suit == Card.HEARTS or (card.suit == Card.SPADES and card.value == 12)

    @staticmethod
    def count_points(deck):
        points = 0
        for card in deck:
            if Hearts.is_point_card( card ):
                points += 1
                if card.suit == Card.SPADES:
                    points += 12
        return points

    def _deal_cards(self):
        """Deal a round of cards to all the players."""
        for p in self.players:
            assert len(p) == 0
        d = Deck()
        d.initialize()
        d.shuffle()
        for i_round in range(13):
            for ip, p in enumerate(self.players):
                p.append(d.pop())
        assert len(d) == 0

    def _player_index(self, card):
        """Find which player is holding a specific card."""
        for ip, p in enumerate(self.players):
            if card in p:
                return ip

    def reset(self):
        self.players = [Player(id) for id in range(4)]

    def play_game(self):
        player_points = np.array([ x.points for x in self.players ])
        while ( player_points < 100 ).all():
            self.play_round()
            player_points = np.array([ x.points for x in self.players ])
        logging.info("Winner of the game is player P{player}".format( player=np.argmax(player_points)))

    def play_round(self):
        self._deal_cards()

        # Pass cards
        hearts_broken = False
        played_cards = [ Deck() for x in self.players ]
        leading_player = self._player_index(Card(2, Card.CLUBS))
        for round in range(13):
            logging.debug("=========== ROUND {round} =========".format(round=round))
            trick = Trick( hearts_broken, round == 0 )
            for turn in range(4):
                current_player = ( leading_player + turn ) % 4
                possible_plays = trick.valid_plays(self.players[ current_player ])
                if not possible_plays:
                    logging.error("Unable to find a possible play:")
                    logging.error("{trick}".format(trick=trick))
                    logging.error("Player cards: {cards}".format(cards=self.players[current_player].tostring()))
                    raise RuntimeError("Unable to find a possible play!")

                # Determine which card to play (randomly!!!)
                c = random.choice( possible_plays )
                logging.debug( "[Round {round}] {trick: <18}. P{player} possible plays: {plays: <50} .... Chose: {card}".format(
                    trick=trick, plays=possible_plays.tostring(), player=current_player, card=c, round=round))
                self.players[ current_player ].remove( c )
                trick.append( c )

            hearts_broken = trick.hearts_broken
            winning_player = (trick.winner() + leading_player) % 4
            logging.info( "[Round {round}] {trick: <18}  WINNER: {winner} (P{player})".format(
                round=round, trick=trick, player=winning_player, winner=trick[trick.winner()]))
            leading_player = winning_player
            for c in trick:
                played_cards[winning_player].append(c)

        #Tally points
        points = np.array(map( Hearts.count_points, played_cards ))
        logging.info( "Points: {points}".format( points = points ))
        if any(points == 26):
            points = -points + 26
        for point, player in zip( points, self.players ):
            player.points += point

class HeartsState():
    def __init__(self):
        self.reset()

    @staticmethod
    def is_point_card(card):
        return card.suit == Card.HEARTS or (card.suit == Card.SPADES and card.value == 12)

    @staticmethod
    def count_points(deck):
        points = 0
        for card in deck:
            if Hearts.is_point_card( card ):
                points += 1
                if card.suit == Card.SPADES:
                    points += 12
        return points

    def _deal_cards(self):
        """Deal a round of cards to all the players."""
        for p in self.players:
            assert len(p) == 0
        d = Deck()
        d.initialize()
        d.shuffle()
        for i_round in range(13):
            for ip, p in enumerate(self.players):
                p.append(d.pop())
        assert len(d) == 0

    def _player_index(self, card):
        """Find which player is holding a specific card."""
        for ip, p in enumerate(self.players):
            if card in p:
                return ip

    def reset(self):
        self.players = [Player(id) for id in range(4)]
        self.played_cards = [ Deck() for x in self.players ]
        self._deal_cards()
        self.round = 0
        self.turn = -1
        self.hearts_broken = False
        self.trick = Trick( self.hearts_broken, self.round == 0 )

        self.leading_player = (self._player_index( Card(2, Card.CLUBS))) % 4
        self.playerJustMoved = self.leading_player

    def Clone(self):
        """Create a deep copy of the HeartsState"""
        st = HeartsState()
        st.players = [ p.copy() for p in self.players ]
        st.played_cards = [ d.copy() for d in self.played_cards ]
        st.round = self.round
        st.turn = self.turn
        st.hearts_broken = self.hearts_broken
        st.trick = self.trick.copy()
        st.leading_player = self.leading_player
        st.playerJustMoved = self.playerJustMoved
        return st

    def DoMove(self, card ):
        self.turn += 1
        self.playerJustMoved = ( self.leading_player + self.turn ) % 4
        possible_plays = self.trick.valid_plays(self.players[ self.playerJustMoved ])
        logging.debug( "[Round {round}] {trick: <18}. P{player} possible plays: {plays: <50} .... Chose: {card}".format(
            trick=self.trick, player=self.playerJustMoved, plays=possible_plays, card=card, round=self.round))
        self.players[ self.playerJustMoved ].remove( card )
        self.trick.append( card )

        if len(self.trick) == 4:
            self.hearts_broken = self.trick.hearts_broken
            winning_player = ( self.trick.winner() + self.leading_player ) % 4
            logging.info( "[Round {round}] {trick: <18}  WINNER: {winner} (P{player})".format(
                round=self.round, trick=self.trick, player=winning_player, winner=self.trick[self.trick.winner()]))
            self.leading_player = winning_player
            for c in self.trick:
                self.played_cards[ winning_player ].append( c )
            self.round += 1
            self.turn = -1
            self.trick = Trick(self.hearts_broken, self.round == 0)

    def GetMoves(self):
        next_player = ( self.leading_player + self.turn + 1) % 4
        possible_plays = self.trick.valid_plays(self.players[ next_player ])
        return possible_plays

    def GetResult(self, player_index ):
        points = np.array(map( Hearts.count_points, self.played_cards ))
        if any(points == 26):
            points = -points + 26
        win_value = points == points.min()
        win_value = win_value.astype('float') / win_value.sum()
        return win_value[player_index]

    def __repr__(self):
        next_player = ( self.leading_player + self.turn + 1) % 4
        return "[Round {round}].\n{trick}\n{player}".format(
            round=self.round, trick=self.trick, player=self.players[next_player] )

def UCTPlayHearts():
    state = HeartsState()
    while (state.GetMoves() != []):
        print str(state)
        if (state.leading_player + state.turn)%4 == 0:
            m = UCT.UCT(rootstate = state, itermax = 1000, verbose = False) # play with values for itermax and verbose = True
        else:
            m = UCT.UCT(rootstate = state, itermax = 100, verbose = False)
        print "Best Move: " + str(m) + "\n"
        logging.root.setLevel(logging.INFO)
        state.DoMove(m)
        logging.root.setLevel(logging.WARNING)
    if state.GetResult(state.playerJustMoved) == 1.0:
        print "Player " + str(state.playerJustMoved) + " wins!"
    elif state.GetResult(state.playerJustMoved) == 0.0:
        print "Player " + str(3 - state.playerJustMoved) + " wins!"
    else: print "Nobody wins!"

def PUCTPlayHearts():
    state = HeartsState()
    while (state.GetMoves() != []):
        print str(state)
        if (state.leading_player + state.turn)%4 == 0:
            m = UCT.PUCT(rootstate = state, itermax = 1000, verbose = False) # play with values for itermax and verbose = True
        else:
            m = UCT.PUCT(rootstate = state, itermax = 100, verbose = False)
        print "Best Move: " + str(m) + "\n"
        logging.root.setLevel(logging.INFO)
        state.DoMove(m)
        logging.root.setLevel(logging.WARNING)
    if state.GetResult(state.playerJustMoved) == 1.0:
        print "Player " + str(state.playerJustMoved) + " wins!"
    elif state.GetResult(state.playerJustMoved) == 0.0:
        print "Player " + str(3 - state.playerJustMoved) + " wins!"
    else: print "Nobody wins!"