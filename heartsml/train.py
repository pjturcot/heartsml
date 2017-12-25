import core
import UCT
import logging
import keras
import collections
import numpy as np
import random

GameMoves = collections.namedtuple( "GameMoves", [ "state", "pi", "moves", "value"])

class HeartsNet():

    def __init__(self):
        self.model = self.build_net()

    def extract_state_feature(self, state):
        trick_array = state.trick.asarray()   # (4, 52) binary vector
        current_hand = state.players[ state.current_player() ].hand.asarray()
        all_cards_played = state.all_cards_played()
        feature = np.vstack( (current_hand, trick_array, all_cards_played ) )
        return feature.transpose()

    def build_net(self, regularizer=keras.regularizers.l2(1e-4)):
        """Build the neural network."""
        s = core.HeartsState()
        feature = self.extract_state_feature( s )

        input = keras.layers.Input( feature.shape , name="input" )
        x = keras.layers.Conv1D(32, 1, kernel_regularizer=regularizer, name='shared_conv1')(input)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv1D(64, 1, kernel_regularizer=regularizer, name='shared_conv2')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv1D(64, 1, kernel_regularizer=regularizer, name='shared_conv3')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        y_action = keras.layers.Conv1D(64, 1, kernel_regularizer=regularizer, name='action_conv4')(x)
        y_action = keras.layers.BatchNormalization()(y_action)
        y_action = keras.layers.Activation('relu')(y_action)
        y_action = keras.layers.Flatten()(y_action)
        y_action = keras.layers.Dense(52, kernel_regularizer=regularizer, name='action_dense5')(y_action)
        y_action = keras.layers.Activation('softmax', name="action")(y_action)
        y_value = keras.layers.Conv1D(1, 1, kernel_regularizer=regularizer, name='value_conv4')(x)
        y_value = keras.layers.BatchNormalization()(y_value)
        y_value = keras.layers.Activation('relu')(y_value)
        y_value = keras.layers.Dense(128, kernel_regularizer=regularizer, name='value_dense5')(y_value)
        y_value = keras.layers.Activation('relu')(y_value)
        y_value = keras.layers.Flatten()(y_value)
        y_value = keras.layers.Dense(1, kernel_regularizer=regularizer, name='value_dense6')(y_value)
        y_value = keras.layers.Activation('sigmoid', name="value")(y_value)


        m = keras.models.Model( [ input ], [ y_action, y_value ] )
        sgd = keras.optimizers.SGD(lr=0.02, momentum=0.9)
        m.compile( optimizer=sgd, loss={"action": "binary_crossentropy", "value": "mean_squared_error"} )
        return m

    def train_on_batch(self, *args, **kwargs ):
        return self.model.train_on_batch( *args, **kwargs )

    def predict(self, state ):
        feature = self.extract_state_feature( state )
        outputs = self.model.predict( {'input': np.array([feature]) } )
        return (outputs[0], outputs[1])


class HeartsTrainer():
    """Class to play a game of hearts using the PUCT algorithm while keeping track of state variables."""
    def __init__(self, heartsnet, mcts_iter_max=100, max_score=50):
        self.state = None
        self.net = heartsnet
        self.mcts_iter_max=mcts_iter_max
        self.game_moves = []

        self.max_score = max_score

    def train(self, n_games=100, n_iters_per_game=10, T=0):
        for i in range(n_games):
            print "**************************************************************************"
            print "Playing game {i} of {n_games}".format( i=i+1, n_games=n_games)
            print "**************************************************************************"
            self.play_game(T=T)
            self.train_net( n_iters_per_game, batch_size=100)

    def play_game(self, T=0):
        self.state = core.HeartsState(max_score=self.max_score)
        game_moves = []

        while self.state.GetMoves() != []:
            if self.state.round() == 0 and self.state.turn == -1:
                node = UCT.Node(state=self.state, net=self.net)
            print str(self.state)
            m, node, moves, PI = UCT.PUCT(rootnode=node, rootstate=self.state, itermax=self.mcts_iter_max, verbose=False, T=T, net=self.net)  # play with values for itermax and verbose = True
            print "Best Move: " + str(m) + "\n"
            game_moves.append( GameMoves( state=self.state.Clone(), moves=moves, pi=PI, value=None))
            logging.root.setLevel(logging.INFO)
            self.state.DoMove(m)
            logging.root.setLevel(logging.WARNING)
        for g in game_moves:
            self.game_moves.append( GameMoves( state=g.state, moves=g.moves, pi=g.pi, value=self.state.GetResult(g.state.current_player())))

    def train_net(self, n_iterations, batch_size=8 ):

        for i in range(n_iterations):
            x_input = []
            y_action = []
            y_value = []
            for i in range(batch_size):
                m = random.choice( self.game_moves )
                action_probs = np.zeros((52,))
                action_probs[ [c.index for c in m.moves] ] = m.pi
                features = self.net.extract_state_feature( m.state )
                m.state.current_player()
                x_input.append(features)
                y_action.append( action_probs )
                y_value.append( m.value )
            inputs = { 'input' : np.array(x_input) }
            outputs = { 'action': np.array(y_action),
                        'value': np.array(y_value) }
            self.net.train_on_batch( inputs, outputs )

    def predict(self, state ):
        return self.net.predict( state )

