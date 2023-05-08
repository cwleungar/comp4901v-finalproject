import os
import argparse

def convert_kitti_to_mot(kitti_dir, mot_dir):
    if not os.path.exists(mot_dir):
        os.makedirs(mot_dir)

    for filename in os.listdir(kitti_dir):
        if filename.endswith('.txt') and 'tracklet' not in filename:
            kitti_file = os.path.join(kitti_dir, filename)
            mot_file = os.path.join(mot_dir, filename.replace('.txt', '.txt'))

            with open(kitti_file, 'r') as f_in, open(mot_file, 'w') as f_out:
                for line in f_in:
                    line = line.strip().split()
                    frame_id = int(line[0])
                    obj_id = int(line[1])
                    if obj_id == -1:
                        continue
                    x, y, w, h = map(float, line[6:10])
                    conf = -1  # KITTI does not provide detection confidence scores
                    cls = -1   # KITTI does not provide object class labels
                    visibility = -1  # KITTI does not provide object visibility information
                    ignored = -1  # KITTI does not provide information on whether an object should be ignored

                    mot_line = '{},{},{:.2f},{:.2f},{:.2f},{:.2f},{},{},{},{}\n'.format(frame_id, obj_id, x, y, w, h, conf, cls, visibility, ignored)
                    f_out.write(mot_line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kitti_dir', type=str, required=True, help='Path to KITTI label directory')
    parser.add_argument('--mot_dir', type=str, required=True, help='Path to output MOT label directory')
    args = parser.parse_args()

    convert_kitti_to_mot(args.kitti_dir, args.mot_dir)