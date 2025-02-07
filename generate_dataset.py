import os
import requests
import argparse


def download_dataset(url: str, save_path: str, max_chars: int = None):
    """Download dataset from the given URL and save it to the specified path."""
    print(f'>>> Fetching dataset from {url}...')
    
    response = requests.get(url, allow_redirects=True)
    response.raise_for_status()
    
    data = response.content if max_chars is None else response.content[:max_chars]
    
    with open(save_path, 'wb') as f:
        f.write(data)
    
    print(f'>>> Dataset saved in {save_path}')


def main():
    parser = argparse.ArgumentParser(description='Download and save a dataset.')
    parser.add_argument('--url', required=False, type=str, default='https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt', help='URL to download the dataset')
    parser.add_argument('--max_chars', type=int, default=None, help='Maximum characters to save (optional)')
    parser.add_argument('--save_dir', type=str, default='data', help='Directory to save the dataset (default: data)')
    parser.add_argument('--save_name', type=str, default='input.txt', help='Filename to save as (default: input.txt)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        print(f'>>> Creating directory {args.save_dir}...')
        os.makedirs(args.save_dir)
    
    save_path = os.path.join(args.save_dir, args.save_name)
    download_dataset(args.url, save_path, args.max_chars)


if __name__ == '__main__':
    main()