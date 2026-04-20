import argparse
import subprocess

def run_adjoin_code(train_file_path, stream_file_path, is_train=True):
    if is_train:
        print('Training the model ...')
        subprocess.run(['python', train_file_path], check=True)
    else:
        print('loaded trained logs')
        
    subprocess.run(['streamlit', 'run', stream_file_path])
        
if __name__ == '__main__':
    print('__loaded__runner.py__')
    parse = argparse.ArgumentParser(description="Adjoin-train-debbug")
    
    parse.add_argument('--train_file_path', type=str, default='model/sample_train_2.py', help='model-train-module')
    parse.add_argument('--stream_file_path', type=str, default='main_stream.py', help='debbug-stream')
    
    args = parse.parse_args()
    
    run_adjoin_code(args.train_file_path, args.stream_file_path)