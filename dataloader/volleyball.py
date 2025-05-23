# ------------------------------------------------------------------------
# Reference:
# https://github.com/cvlab-epfl/social-scene-understanding/blob/master/volleyball.py
# https://github.com/wjchaoGit/Group-Activity-Recognition/blob/master/volleyball.py
# ------------------------------------------------------------------------
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np
import random
from PIL import Image
import re
import os.path as osp

ACTIVITIES = ['r_set', 'r_spike', 'r-pass', 'r_winpoint', 'l_set', 'l-spike', 'l-pass', 'l_winpoint']
ls = ['ret','left','set','spike','pass','winpoint']


def volleyball_read_annotations(path, seqs, num_activities):
    labels = {}
    if num_activities == 8:
        group_to_id = {name: i for i, name in enumerate(ACTIVITIES)}
    # merge pass/set label    
    elif num_activities == 6:
        group_to_id = {'r_set': 0, 'r_spike': 1, 'r-pass': 0, 'r_winpoint': 2,
                       'l_set': 3, 'l-spike': 4, 'l-pass': 3, 'l_winpoint': 5}
    
    for sid in seqs:
        annotations = {}
        with open(path + '/%d/annotations.txt' % sid) as f:
            for line in f.readlines():
                values = line[:-1].split(' ')
                file_name = values[0]
                fid = int(file_name.split('.')[0])

                activity = group_to_id[values[1]]

                annotations[fid] = {
                    'file_name': file_name,
                    'group_activity': activity,
                }
            labels[sid] = annotations

    return labels


def volleyball_all_frames(labels):
    frames = []

    for sid, anns in labels.items():
        for fid, ann in anns.items():
            frames.append((sid, fid))

    return frames


class VolleyballDataset(data.Dataset):
    """
    Volleyball Dataset for PyTorch
    """
    def __init__(self, frames, anns, image_path, args, is_training=True):
        super(VolleyballDataset, self).__init__()
        self.frames = frames
        self.anns = anns
        self.image_path = image_path
        self.image_size = (args.image_width, args.image_height)
        self.random_sampling = args.random_sampling
        self.num_frame = args.num_frame
        self.num_total_frame = args.num_total_frame
        self.is_training = is_training
        self.transform = transforms.Compose([
            transforms.Resize((args.image_height, args.image_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    def __getitem__(self, idx):
        frames = self.select_frames(self.frames[idx])
        samples = self.load_samples(frames)

        return samples

    def __len__(self):
        a =len(self.frames)
        return len(self.frames)

    def select_frames(self, frame):
        """
        Select one or more frames
        """
        sid, src_fid = frame

        if self.is_training:
            if self.random_sampling:
                sample_frames = random.sample(range(src_fid - 5, src_fid + 5), self.num_frame)
                sample_frames.sort()
            else:
                segment_duration = self.num_total_frame // self.num_frame
                sample_frames = np.multiply(list(range(self.num_frame)), segment_duration) + np.random.randint(
                    segment_duration, size=self.num_frame) + src_fid - segment_duration * (self.num_frame // 2)
        else:
            segment_duration = self.num_total_frame // self.num_frame
            sample_frames = np.multiply(list(range(self.num_frame)), segment_duration) + src_fid - segment_duration * (self.num_frame // 2)

        return [(sid, src_fid, fid) for fid in sample_frames]

    def load_samples(self, frames):
        ls, images, activities =[],  [], []
        w = np.random.rand()
        if self.is_training and w >0.5:
            flip = True
        else:
            flip = False

        for i, (sid, src_fid, fid) in enumerate(frames):
            img = Image.open(self.image_path + '/%d/%d/%d.jpg' % (sid, src_fid, fid))

            if flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = self.transform(img)

            images.append(img)
            c = (sid, src_fid, fid)
            ls.append(c)
            activitie = self.anns[sid][src_fid]['group_activity']
            if flip:
                activitie = (activitie + 4) % 8
            activities.append(activitie)

        # if activities[0] == 0:   labels = [1, 0, 1, 0, 0, 0]
        # elif activities[0] == 1: labels = [1, 0, 0, 1, 0, 0]
        # elif activities[0] == 2: labels = [1, 0, 0, 0, 1, 0]
        # elif activities[0] == 3: labels = [1, 0, 0, 0, 0, 1]
        # elif activities[0] == 4: labels = [0, 1, 1, 0, 0, 0]
        # elif activities[0] == 5: labels = [0, 1, 0, 1, 0, 0]
        # elif activities[0] == 6: labels = [0, 1, 0, 0, 1, 0]
        # elif activities[0] == 7: labels = [0, 1, 0, 0, 0, 1]

        if activities[0] == 0:
            labels = [1, 0, 1, 0, 0, 0]
            label0 = [0]
            label1 = [0]
        elif activities[0] == 1:
            labels = [1, 0, 0, 1, 0, 0]
            label0 = [0]
            label1 = [1]
        elif activities[0] == 2:
            labels = [1, 0, 0, 0, 1, 0]
            label0 = [0]
            label1 = [2]
        elif activities[0] == 3:
            labels = [1, 0, 0, 0, 0, 1]
            label0 = [0]
            label1 = [3]
        elif activities[0] == 4:
            labels = [0, 1, 1, 0, 0, 0]
            label0 = [1]
            label1 = [0]
        elif activities[0] == 5:
            labels = [0, 1, 0, 1, 0, 0]
            label0 = [1]
            label1 = [1]
        elif activities[0] == 6:
            labels = [0, 1, 0, 0, 1, 0]
            label0 = [1]
            label1 = [2]
        elif activities[0] == 7:
            labels = [0, 1, 0, 0, 0, 1]
            label0 = [1]
            label1 = [3]

        # if activities[0] == 0:   labels = [1, 1, 0, 0, 0]
        # elif activities[0] == 1: labels = [1, 0, 1, 0, 0]
        # elif activities[0] == 2: labels = [1, 0, 0, 1, 0]
        # elif activities[0] == 3: labels = [1, 0, 0, 0, 1]
        # elif activities[0] == 4: labels = [0, 1, 0, 0, 0]
        # elif activities[0] == 5: labels = [0, 0, 1, 0, 0]
        # elif activities[0] == 6: labels = [0, 0, 0, 1, 0]
        # elif activities[0] == 7: labels = [0, 0, 0, 0, 1]

        images = torch.stack(images)
        # ls = torch.stack(ls)
        activities = np.array(activities, dtype=np.int32)
        labels = np.array(labels, dtype=np.int32)
        label0 = np.array(label0, dtype=np.int32)
        label1 = np.array(label1, dtype=np.int32)
        # labels = torch.Tensor(ACTIVITIES, dtype=torch.long)


        # convert to pytorch tensor
        activities = torch.from_numpy(activities).long()
        labels = torch.from_numpy(labels).long()
        label0 = torch.from_numpy(label0).long()
        label1 = torch.from_numpy(label1).long()


        return  ls, images, activities, labels, label0, label1

class VolleyballDataset1(data.Dataset):
    """
    Volleyball Dataset for PyTorch
    """
    def __init__(self, frames, anns, image_path, args, is_training=True):
        super(VolleyballDataset, self).__init__()
        self.frames = frames
        self.anns = anns
        self.image_path = image_path
        self.image_size = (args.image_width, args.image_height)
        self.random_sampling = args.random_sampling
        self.num_frame = args.num_frame
        self.num_total_frame = args.num_total_frame
        self.is_training = is_training
        self.transform = transforms.Compose([
            transforms.Resize((args.image_height, args.image_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, idx):
        frames = self.select_frames(self.frames[idx])
        samples = self.load_samples(frames)

        return samples

    def __len__(self):
        a =len(self.frames)
        return len(self.frames)

    def select_frames(self, frame):
        """
        Select one or more frames
        """
        sid, src_fid = frame

        if self.is_training:
            if self.random_sampling:
                sample_frames = random.sample(range(src_fid - 5, src_fid + 5), self.num_frame)
                sample_frames.sort()
            else:
                segment_duration = self.num_total_frame // self.num_frame
                sample_frames = np.multiply(list(range(self.num_frame)), segment_duration) + np.random.randint(
                    segment_duration, size=self.num_frame) + src_fid - segment_duration * (self.num_frame // 2)
        else:
            segment_duration = self.num_total_frame // self.num_frame
            sample_frames = np.multiply(list(range(self.num_frame)), segment_duration) + src_fid - segment_duration * (self.num_frame // 2)

        return [(sid, src_fid, fid) for fid in sample_frames]

    def load_samples(self, frames):
        ls, images, activities =[],  [], []
        ###################
        # pattern = re.compile(r'([-\d]+)_c([-\d]+)')
        pattern = re.compile(r'([-\d]+)')
        all_pids = {}
        idx2pid = []
        ret = []
        cnt = 0
        ####################

        for i, (sid, src_fid, fid) in enumerate(frames):
            img = Image.open(self.image_path + '/%d/%d/%d.jpg' % (sid, src_fid, fid))

            img = self.transform(img)

            images.append(img)
            c = (sid, src_fid, fid)
            ls.append(c)
            activities.append(self.anns[sid][src_fid]['group_activity'])

            #######################
            fname = osp.basename(self.image_path + '/%d/%d/%d.jpg' % (sid, src_fid, fid))
            pid= map(int, pattern.search(fname).groups())#, cam
            if pid == -1: continue  # junk images are just ignored
            if i == 1:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            pid = all_pids[pid]
            # cam -= 1
            # ret.append( cnt)
            idx2pid.append(pid)
            # cnt = cnt +1
            #################

        # ret = torch.stack(ret)
        # idx2pid = torch.stack(idx2pid)
        ret.append(cnt)
        images = torch.stack(images)
        # ls = torch.stack(ls)
        activities = np.array(activities, dtype=np.int32)
        ret = np.array(ret, dtype=np.int32)

        # convert to pytorch tensor
        activities = torch.from_numpy(activities).long()
        ret = torch.from_numpy(ret).long()

        return  ls, images, activities, ret
