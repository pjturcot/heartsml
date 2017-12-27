import core
import UCT
import logging
import keras
import collections
import numpy as np
import random

GameMoves = collections.namedtuple( "GameMoves", [ "state", "pi", "moves", "value"])

class HeartsNet():

    def __init__(self, result_type='points', value_activation='tanh', lr=0.2):
        self.value_activation = value_activation
        self.result_type = result_type
        self.lr = lr
        self.model = self.build_net()

    def extract_state_feature(self, state):
        player_order = (np.arange(4)+state.current_player()) % 4
        trick_array = state.trick.asarray()   # (4, 52) binary vector
        current_hand = state.players[ state.current_player() ].hand.asarray()
        cards_played_by_player = np.array([p.cards_won.asarray() for p in state.players])[ player_order ]
        feature = np.vstack( (current_hand, trick_array, cards_played_by_player ) )
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
        if self.value_activation is None:
            y_value = keras.layers.Dense(1, kernel_regularizer=regularizer, name='value')(y_value)
        else:
            y_value = keras.layers.Dense(1, kernel_regularizer=regularizer, name='value_dense6')(y_value)
            y_value = keras.layers.Activation(self.value_activation, name="value")(y_value)

        m = keras.models.Model( [ input ], [ y_action, y_value ] )
        sgd = keras.optimizers.SGD(lr=self.lr, momentum=0.9)
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
    def __init__(self, heartsnet, mcts_iter_max=100, max_score=50, mcts_c_puct=1.0):
        self.state = None
        self.net = heartsnet
        self.result_type = heartsnet.result_type
        self.mcts_iter_max=mcts_iter_max
        self.mcts_c_puct=mcts_c_puct
        self.game_moves = []
        self.max_score = max_score

    def train(self, n_games=100, n_iters_per_game=10, T=0, batch_size=100):
        for i in range(n_games):
            print "**************************************************************************"
            print "Playing game {i} of {n_games}".format( i=i+1, n_games=n_games)
            print "**************************************************************************"
            self.play_game(T=T)
            print "**************************************************************************"
            for i in range(4):
                print self.state.players[i], " got result: ", self.state.GetResult(i, result_type=self.result_type)
            print "**************************************************************************"
            print "Training net..."
            self.train_net( n_iters_per_game, batch_size=batch_size)

    def play_game(self, T=0):
        self.state = core.HeartsState(max_score=self.max_score)
        game_moves = []
        node = AlphaZeroNode( state=self.state, net=self.net )
        while self.state.GetMoves() != []:
            node.RunSimulations( self.state, n_simulations=self.mcts_iter_max, c_puct=self.mcts_c_puct )
            print str(self.state)
            edge, moves, PI = node.GetMCTSResult(T=T)
            for e in node.edges:
                print e
            m, node = edge.action, edge.child
            print "Best Move: " + str(m) + "\n"
            game_moves.append( GameMoves( state=self.state.Clone(), moves=moves, pi=PI, value=None))
            logging.root.setLevel(logging.INFO)
            self.state.DoMove(m)
            logging.root.setLevel(logging.WARNING)
        for g in game_moves:
            self.game_moves.append(
                GameMoves( state=g.state, moves=g.moves, pi=g.pi,
                           value=self.state.GetResult(g.state.current_player(),result_type=self.result_type)))

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
                x_input.append(features)
                y_action.append( action_probs )
                y_value.append( m.value )
            inputs = { 'input' : np.array(x_input) }
            outputs = { 'action': np.array(y_action),
                        'value': np.array(y_value) }
            self.net.train_on_batch( inputs, outputs )

    def predict(self, state ):
        return self.net.predict( state )


class AlphaZeroEdge:

    def __init__(self, action=None, prior=None, parent=None, child=None, value=0.0 ):
        self.action = action
        self.prior = prior

        self.parent = parent
        self.child = child

        self.visits = 0   # MCTS algorithm tracking visits through this edge
        self.total_action_value = 0.0   #
        self.mean_action_value = value

    def __repr__(self):
        return "Action:{action} (N={visits}, Q={mean_value}, T={total_value}, P={prior}".format(
            action=self.action, visits=self.visits, mean_value=self.mean_action_value, total_value=self.total_action_value, prior=self.prior)

    def get_child(self):
        """Get the node encoded by this action."""
        if self.child is None:
            print "Initializing child state for edge: {edge}".format( edge=self )
            s = self.parent.state.Clone()
            s.DoMove( self.action )
            self.child = AlphaZeroNode( state=s, net=self.parent.net )
        else:
            print "Returning existing node: {edge}".format( edge=self )
        return self.child


class AlphaZeroNode:

    def __init__(self, state=None, net=None):
        self.state = state
        self.net = net
        self.probs, self.value = self.net.predict( state )
        self.probs = self.probs.flatten()
        self.value = float(self.value)

        self.actions = self.state.GetMoves()
        self.edges = None

    def is_leaf(self):
        return len(self.actions) == 0

    def InitEdges(self):
        """Initialize all the actions from the current state."""
        if self.edges is None:
            self.edges = []
            prior_total = sum( [self.probs[a.index] for a in self.actions] )
            for a in self.actions:
                self.edges.append( AlphaZeroEdge( action=a, prior=self.probs[a.index]/prior_total, parent=self, child=None ))


    def PUCTSelectChild(self, C_PUCT=26.0, dirichlet=None):
        """ Use the PUCT formula to select a child node. Often a constant UCTK is applied so we have
            lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
            exploration versus exploitation.
        """
        action_score = self.get_action_scores( C_PUCT=C_PUCT, dirichlet=dirichlet )
        i = action_score.argmax()
        return self.edges[i]

    def get_action_scores(self, C_PUCT=26.0, dirichlet=None ):
        N = np.array( [ e.visits for e in self.edges ] )
        U_num = np.sqrt(N.sum())
        Q = np.array([ e.mean_action_value for e in self.edges ])
        P = np.array([ e.prior for e in self.edges ] )
        if dirichlet is not None:
            P += np.random.dirichlet( [dirichlet]*52 , 1 )[0, :P.size]
        action_score = Q + C_PUCT * P * U_num / (1 + N )
        return action_score

    def RunSimulations(self, root_state, n_simulations=None, n_max_simulations=None, c_puct=1.0):
        """Run branching simulations from this node starting as the root node."""
        assert self.actions == root_state.GetMoves()
        self.InitEdges()
        n_current_simulations = sum( [ e.visits for e in self.edges ] )
        if n_max_simulations and n_simulations is None:
            n_simulations = n_max_simulations - n_current_simulations

        for i_simulation in range(n_simulations):
            state = root_state.Clone()
            edges_traversed = []
            node = self
            while not node.is_leaf():
                if node == self:
                    e = node.PUCTSelectChild( dirichlet=0.9, C_PUCT=c_puct)   # Sample with noise.
                else:
                    e = node.PUCTSelectChild( C_PUCT=c_puct)
                edges_traversed.append( e )
                state.DoMove(e.action)
                if e.child is None:
                    node = AlphaZeroNode( state=state, net=node.net )
                    node.InitEdges()
                    e.child = node
                else:
                    node = e.child
            if e.child is None:
                node = AlphaZeroNode( state=state, net=node.net )
                node.InitEdges()
                e.child = node

            # We've reached the end
            value = None
            for e in edges_traversed[::-1]:
                if value is None:
                    value = e.child.value
                    logging.debug( "Found value: {value}".format(value=value) )
                e.visits += 1
                e.total_action_value += value
                e.mean_action_value = e.total_action_value / e.visits
                logging.debug( e )

    def GetMCTSResult(self, T=0.0):
        if T > 0:
            N_a = np.array([e.visits for e in self.edges], dtype='float') ** (1 / T)
            PI = N_a / np.sum(N_a)
            edge = np.random.choice( self.edges , p=PI )
        elif T == 0:
            N = np.array([e.visits for e in self.edges])
            ix = N.argmax()
            edge = self.edges[ix]
            PI = np.zeros( (len(self.edges)))
            PI[ix] = 1.0
        return edge, self.actions, PI






