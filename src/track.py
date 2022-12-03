from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch
from PIL import Image

from tracker.multitracker import JDETracker, STrack
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets

from tracking_utils.utils import mkdir_if_missing
from opts import opts


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_score(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},{s},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, s=score)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30, use_cuda=True):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    prev_online_targets = []
    prev_img = None
    eigen_threshold = int(opt.eigen_threshold)
    detect_frame_interval = int(opt.detect_frame_interval)
    num_detect = 0
    num_skipped = 0
    prev_area = 0
    num_consecutive_skips = 0
    total_areas = []
    largest_areas = []
    #for path, img, img0 in dataloader:
    for i, (path, img, img0) in enumerate(dataloader):
        #if i % 8 != 0:
            #continue   
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        


        # run tracking
        timer.tic()

        if i > 0 :
            total_eig = 0
            num_boxes_counted = 0
            for prev_track in prev_online_targets:
                #filter targets like below
                previous_position_tlbr = prev_track.tlbr
                predicted_curr_position_tlbr = prev_track.predict_tlbr_without_updating_state()
                prev_detected_box, curr_predicted_box = get_crop_image_same_size(prev_img, previous_position_tlbr, img0, predicted_curr_position_tlbr)
                
                prev_tlwh = prev_track.tlwh
                vertical = prev_tlwh[2] / prev_tlwh[3] > 1.6
                if prev_tlwh[2] * prev_tlwh[3] > opt.min_box_area and not vertical:
                    total_eig += compute_eigen_value_similarity(prev_detected_box, curr_predicted_box)
                    num_boxes_counted += 1
                    #print('index', i, previous_position_tlbr, predicted_curr_position_tlbr)
            print('eig_', i ,": ", total_eig, 'num_boxes_counted ', num_boxes_counted)
        else:
            total_eig = 10000

        if use_cuda:
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
        else:
            blob = torch.from_numpy(img).unsqueeze(0)

        if total_eig >= eigen_threshold or num_consecutive_skips >=  detect_frame_interval:
          online_targets = tracker.update(blob, img0)
          prev_online_targets = online_targets
          prev_img = img0
          num_detect+=1
          num_consecutive_skips = 0
          print('detect at ', i, ' prev_area: ', prev_area)
        else:
          #eig = compute_eigen_values_consecutive(prev_img, img0)
          STrack.multi_predict(prev_online_targets)
          online_targets = prev_online_targets
          num_consecutive_skips += 1
          num_skipped+=1
        online_tlwhs = []
        online_ids = []
        #online_scores = []
        tot_area = 0
        max_area = -1
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                curr_area = tlwh[2] * tlwh[3]
                tot_area += curr_area
                #online_scores.append(t.score)
                if curr_area > max_area:
                    max_area = curr_area

        prev_area = tot_area
        largest_areas.append(max_area)
        total_areas.append(tot_area)
        timer.toc()
        #print('largest_areas:', largest_areas)
        #print('total_areas:', total_areas)
        
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        #results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    # save results
    print('num_detect:', num_detect, "num_skipped:", num_skipped)
    write_results(result_filename, results, data_type)
    #write_results_score(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls

def get_image_as_array(img):
    image_1 = Image.fromarray(img)
    imgGray = image_1.convert('L')
    img_gray = np.array(imgGray)
    img_part = img_gray
    img_part = img_part.reshape(-1)
    return img_part

def compute_eigen_values_consecutive(image1, image2):
    img1 = get_image_as_array(image1)
    img2 = get_image_as_array(image2)
    cova_1 = np.cov(img1, img2)
    eig_1, eig_vec_1 =  np.linalg.eig(cova_1)
    eig = np.sort(eig_1)
    return eig[0]

def compute_eigen_value_similarity(img1, img2):
    img1 = img1.reshape(-1)
    img2 = img2.reshape(-1)
    cova_1 = np.cov(img1, img2)
    eig_1, eig_vec_1 =  np.linalg.eig(cova_1)
    eig = np.sort(eig_1)
    return eig[0]

def get_image_crop(img1, boundingbox1):
    image_1 = Image.fromarray(img1)
    imgGray_1 = image_1.convert('L')
    img_crop1 = imgGray_1.crop(boundingbox1)
    return img_crop1

def get_crop_image_same_size(img1, boundingbox1, img2, boundingbox2):
    img_crop1 = get_image_crop(img1, boundingbox1)
    img_crop2 = get_image_crop(img2, boundingbox2)
    img_crop2_resized = img_crop2.resize(img_crop1.size)
    img_crop1 = np.array(img_crop1).reshape(-1)
    img_crop2_resized = np.array(img_crop2_resized).reshape(-1)
    return img_crop1, img_crop2_resized

def crop_detected_portion_of_image(image, tlbr):
    #(min x, min y, max x, max y)
    return image[round(tlbr[1]):round(tlbr[3]), round(tlbr[0]):round(tlbr[2])]

def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo',
         save_images=False, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = os.path.join(data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = opts().init()

    if not opt.val_mot16:
        seqs_str = '''KITTI-13
                      KITTI-17
                      ADL-Rundle-6
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte'''
        #seqs_str = '''TUD-Campus'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    else:
        seqs_str = '''MOT16-02
                      MOT16-04
                      MOT16-05
                      MOT16-09
                      MOT16-10
                      MOT16-11
                      MOT16-13'''
        data_root = os.path.join(opt.data_dir, 'MOT16/train')
    if opt.test_mot16:
        seqs_str = '''MOT16-01
                      MOT16-03
                      MOT16-06
                      MOT16-07
                      MOT16-08
                      MOT16-12
                      MOT16-14'''
        #seqs_str = '''MOT16-01 MOT16-07 MOT16-12 MOT16-14'''
        #seqs_str = '''MOT16-06 MOT16-08'''
        data_root = os.path.join(opt.data_dir, 'MOT16/test')
    if opt.test_mot15:
        seqs_str = '''ADL-Rundle-1
                      ADL-Rundle-3
                      AVG-TownCentre
                      ETH-Crossing
                      ETH-Jelmoli
                      ETH-Linthescher
                      KITTI-16
                      KITTI-19
                      PETS09-S2L2
                      TUD-Crossing
                      Venice-1'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/test')
    if opt.test_mot17:
        seqs_str = '''MOT17-01-SDP
                      MOT17-03-SDP
                      MOT17-06-SDP
                      MOT17-07-SDP
                      MOT17-08-SDP
                      MOT17-12-SDP
                      MOT17-14-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/test')
    if opt.val_mot17:
        seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/train')
    if opt.val_mot15:
        seqs_str = '''Venice-2
                      KITTI-13
                      KITTI-17
                      ETH-Bahnhof
                      ETH-Sunnyday
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte
                      ADL-Rundle-6
                      ADL-Rundle-8
                      ETH-Pedcross2'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    if opt.val_mot20:
        seqs_str = '''MOT20-01
                      MOT20-02
                      MOT20-03
                      MOT20-05
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/train')
    if opt.test_mot20:
        seqs_str = '''MOT20-04
                      MOT20-06
                      MOT20-07
                      MOT20-08
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/test')
    if opt.custom_video:
        seqs_str = opt.seq_name
        data_root = os.path.join(opt.data_dir, opt.data_path)
    seqs = [seq.strip() for seq in seqs_str.split()]

    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name='MOT15_test_samplevideo_'+seqs_str,
         show_image=False,
         save_images=False,
         save_videos=True)
