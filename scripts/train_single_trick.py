import heartsml
import pickle

if __name__ == "__main__":
    net = heartsml.train.HeartsNetResidualBlock(
        n_residual_blocks=10, lr=0.05,
        result_type='all_rawpoints', loss_weights={'action': 1.0, 'value': 0.1  })


    t = heartsml.train.HeartsTrainer(net, mcts_iter_max=1000, max_score=13, mcts_c_puct=0.75*3 )
    t.train( n_games=5000, n_iters_per_game=10, T=0.5, batch_size=2000)

    with open('games.pickle','wb') as fout:
        pickle.dump( t.games, fout )
    with open('training_metrics.pickle','wb') as fout:
        pickle.dump( t.training_metrics, fout )
    t.net.model.save( 'weights.h5' )


