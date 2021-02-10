
import gdown
import argparse
import os.path as osp

data2url = {
    'covtype':
    'https://drive.google.com/uc?id=1ncsZtAy1Q3rWAEyztme4eO37f1JEdUU6',
    'kddcup':
    'https://drive.google.com/uc?id=11ql7MDPBsutV3Rv8AD2vVSP-9_H-IG_H',
    'YTB':
    'https://drive.google.com/uc?id=11S7pIp9uDde-AG9c5gAXYwhl0DzDeUWn',
    'MSCM1':
    'https://drive.google.com/uc?id=1ZBp5fu-PGiNBSnHIf5kV7XRaea2ujdVC',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Download Data sets')
    parser.add_argument('--data', default='covtype', choices=data2url.keys())
    parser.add_argument('--filepath', default='./data/covtype.npy', type=str)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    url = data2url[args.data]
    filepath = args.filepath
    if osp.exists(filepath) and not args.force:
        print('{} already exists.'
              'Modify --filepath or run with --force to overwrite.'.format(
                  filepath))
    else:
        gdown.download(url, filepath, quiet=False)
    print('download data successfully!')
