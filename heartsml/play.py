import heartsml.core
import heartsml.train
import logging
import numpy as np

class SinglePlayerHeartsGame():

    def __init__(self, heartsnet, max_score=100 ):
        self.state = None
        self.net = heartsnet
        self.max_score = max_score

    def get_user_play( self ):
        valid_moves = self.state.GetMoves()
        input = None
        print self.state.players[ self.state.current_player() ].hand
        while True:
            input = raw_input("Enter play for player {id}: ".format(id=self.state.current_player()))
            try:
                input = heartsml.core.string2card(input)
            except ValueError as e:
                "Invalid entry: {input} (error={error})".format( input=input, error=e)
            if input in valid_moves:
                break
            else:
                print "Invalid move: {input} ( valid moves: {valid_moves} )".format( input=input, valid_moves=str(list(valid_moves)))

        return input


    def play_game(self, T=0):
        self.state = heartsml.core.HeartsState(max_score=self.max_score)
        game_moves = []
        while self.state.GetMoves() != []:
            if self.state.current_player() == 0:
                self.state.display()
                m = self.get_user_play()
            else:
                probs, value = self.net.predict( self.state )
                probs = probs.flatten()
                valid_moves = self.state.GetMoves()
                valid_probs = [ probs[x.index] for x in valid_moves ]
                m = valid_moves[np.argsort( valid_probs)[-1]]
                print "Player {id} played : {move}".format(id=self.state.current_player(), move=m )
            self.state.DoMove(m)
            print self.state

        final_game_moves = []
        for g in game_moves:
            final_game_moves.append(
                GameMoves( state=g.state, moves=g.moves, pi=g.pi,
                           value=self.state.GetResult(g.state.current_player(),result_type=self.result_type)))
        self.games.append(final_game_moves)
        self.games = self.games
