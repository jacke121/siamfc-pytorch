from __future__ import absolute_import

from got10k.experiments import *

from siamfc import TrackerSiamFC
import os

if __name__ == '__main__':
    # setup tracker
    net_path = 'model/model.pth'
    # net_path =r'D:\project\track\siamfc-pytorch\pretrained\siamfc_new\model_e50.pth'
    tracker = TrackerSiamFC(net_path=net_path)
    experiments = [
        # ExperimentGOT10k(r'F:\迅雷下载\vot2016',subset='test'),
        ExperimentGOT10k(r'model',subset='test'),
        # ExperimentOTB('data/OTB', version=2015),
        # ExperimentDTB70('data/DTB70'),
        # ExperimentUAV123('data/UAV123', version='UAV20L'),
        # ExperimentNfS('data/nfs', fps=240)
    ]

    # run tracking experiments and report performance
    for i,e in enumerate(experiments):
        print(i)

        print('Running tracker %s on GOT-10k...' % tracker.name)
        e.dataset.return_meta = False

        # loop over the complete dataset
        for s, (img_files, anno) in enumerate(e.dataset):
            seq_name = e.dataset.seq_names[s]
            print('--Sequence %d/%d: %s' % (
                s + 1, len(e.dataset), seq_name))

            # run multiple repetitions for each sequence
            for r in range(e.repetitions):
                # check if the tracker is deterministic
                if r > 0 and tracker.is_deterministic:
                    break
                elif r == 3 and e._check_deterministic(
                        tracker.name, seq_name):
                    print('  Detected a deterministic tracker, ' +
                          'skipping remaining trials.')
                    break
                print(' Repetition: %d' % (r + 1))

                # tracking loop
                boxes, times = tracker.track(
                    img_files, anno[0, :], visualize=True)