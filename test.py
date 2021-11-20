import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='JoVA')
    #parser.add_argument('--beta_value', type=float, default=0.01) #0.01 for both Movie-Lens and Pinterest and 0.001 for Yelp.
    parser.add_argument('--train_epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int,default=256)
    parser.add_argument('--using_bpr', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gpu_id', type=int, default=0,help='0 for NAIS_prod, 1 for NAIS_concat')
  #  parser.add_argument('--decay_epoch_step', type=int, default=50,help="decay the learning rate for each n epochs")
    return parser.parse_args()
