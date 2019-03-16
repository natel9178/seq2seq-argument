import _pickle as pickle
import argparse

def process(pickle, save_file, keep_case):
  with open(save_file+'.src', "a") as src_file:
    with open(save_file+'.tgt', "a") as tgt_file:
      for discussion in pickle:
        for src, tgt in zip(discussion['src'], discussion['tgt']):
          src_str = " ".join(src)
          tgt_str = " ".join(tgt)

          if not keep_case:
            src_str = src_str.lower()
            tgt_str = tgt_str.lower()
          src_file.write(src_str + '\n')
          tgt_file.write(tgt_str + '\n')
      
      

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-train', required=True)
  parser.add_argument('-train_save', required=True)
  parser.add_argument('-valid', required=True)
  parser.add_argument('-valid_save', required=True)
  parser.add_argument('-test', required=True)
  parser.add_argument('-test_save', required=True)
  parser.add_argument('-keep_case', action='store_true')
  opt = parser.parse_args()

  train = pickle.load(open( opt.train, "rb" ))
  process(train, opt.train_save, opt.keep_case)

  valid = pickle.load(open( opt.valid, "rb" ))
  process(valid, opt.valid_save, opt.keep_case)
  
  test = pickle.load(open( opt.test, "rb" ))
  process(test, opt.test_save, opt.keep_case)



if __name__ == '__main__':
    main()