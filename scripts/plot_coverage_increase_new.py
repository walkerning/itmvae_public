# -*- coding: utf-8 -*-
import cPickle
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from plot_tsne_hie import get_tensor

use_z_3d = True

items = [
    {
        "name": "1 class",
        "filename": "./results_increase_new/20news_me/dp_prod_13_prior_3_d1000_rebn/1_labels/modelsavetensors.pkl",
        "kwargs": {}
    },
    {
        "name": "2 class",
        "filename": "./results_increase_new/20news_me/dp_prod_13_prior_3_d1000_rebn/2_labels/modelsavetensors.pkl",
        "kwargs": {}
    },
    {
        "name": "5 class",
        "filename": "./results_increase_new/20news_me/dp_prod_13_prior_3_d1000_rebn/5_labels/modelsavetensors.pkl",
        "kwargs": {}
    },
    {
        "name": "10 class",
        "filename": "./results_increase_new/20news_me/dp_prod_13_prior_3_d1000_rebn/10_labels/modelsavetensors.pkl",
        "kwargs": {}
    },
    {
        "name": "20 class",
        "filename": "./results_increase_new/20news_me/dp_prod_13_prior_3_d1000_rebn/20_labels/modelsavetensors.pkl",
        "kwargs": {}
    }
]

tensors = []

#x = range(40)
#items = sorted(train_dct.items(), key=lambda x: x[0])
#plt.figure()
label_size=8
matplotlib.rcParams['xtick.labelsize'] = label_size 
matplotlib.rcParams['ytick.labelsize'] = label_size 
plt.tick_params(axis='both', which='major', labelsize=label_size)
plt.tick_params(axis='both', which='minor', labelsize=label_size)
plt.figure(figsize=(12,4))
ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133)
# x = [0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] + range(24, 145, 8)
# x = range(50) + range(50, 101, 2) + range(100, 200, 4)
x = range(50) + range(50, 100, 2) + range(100, 200, 4)
#x = range(50)

for item in items:
    tensor = cPickle.load(open(item["filename"]))
    tensors.append(tensor)
    if not use_z_3d:
        kwargs = {"use_ab_pi": True}
        kwargs.update(item["kwargs"])
        ab_pi = get_tensor(tensor["names"], tensor["train"], **kwargs)
    else:
        ab_pi = tensor["train"][tensor['names'].index('z_3d')][0]

    sparsity = np.log(np.mean(np.sort(ab_pi, axis=-1), axis=0) + 1e-10)
    max_counts = np.bincount(np.argmax(ab_pi, axis=-1), minlength=200)
    counts_coverage = sorted(max_counts.astype(np.float32) / max_counts.sum(), reverse=True)

    mean_pi = np.mean(ab_pi, axis=0)
    mean_coverage = sorted(mean_pi, reverse=True)

    counts_coverage = np.cumsum(counts_coverage)
    mean_coverage = np.cumsum(mean_coverage)

    data = [0] + list(counts_coverage[np.array(x[1:])-1])
    ax1.plot(range(len(x)), data, label=item["name"])
    # ax1.set_xscale('log')
    #ax1.semilogx(np.array(range(len(x))) + 1, data, label=item["name"])
    data = [0] + list(mean_coverage[np.array(x[1:])-1])
    ax2.plot(range(len(x)), data, label=item["name"])
    # ax2.set_xscale('log')
    #ax2.semilogx(np.array(range(len(x))) + 1, data, label=item["name"])
    data = list(sparsity[np.array(x)])
    ax3.plot(range(len(x)), data, label=item["name"])

font = {
    'weight' : 'bold',
    'size'   : 14,
}
matplotlib.rc('font', **font)

locs, _ = plt.xticks()
# tick_locs = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
tick_locs = range(0, len(x), 10)
#[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
for ax in [ax1, ax2]:
    plt.sca(ax)
    # import pdb
    # pdb.set_trace()
    # ax.set_xticks(range(0, 200, 10))
    # ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    plt.xticks(tick_locs, [str(x[i]) for i in tick_locs])

    plt.xlabel("Topic Number", fontsize=14)
    plt.ylabel("Coverage", fontsize=14)
    plt.grid()
    ax.legend(loc=4)
ax1.set_title("max coverage")
ax2.set_title("mean coverage")
ax3.set_title("sparsity")
plt.sca(ax3)
plt.grid()
plt.legend(loc=4)
plt.xlabel("Topic Index", fontsize=14)
plt.xticks(tick_locs, [str(x[i]) for i in tick_locs])
plt.ylabel("...", fontsize=14)

plt.tight_layout(pad=0.5, w_pad=0.1, h_pad=0.1)
fname = "./plot_coverage_increase{}.png".format("_usez3d" if use_z_3d else "")
plt.savefig(fname)
print "save to ", fname

# # import re
# # #text = open("results_increase_again/20news_me/dp_prod_ori_bn/1_labels/test.txt.oc").read()
# # #coherences = re.findall("[(0-9.)+]", text)

# # return self.sess.run(self._topic_components_tensor)[effective_dims]

# # print "\n".join(["{:.3f}".format(x) for x in np.array(asses)[np.where(np.array(asses) > 0.005)]])



#plt.legend(loc=4)
#plt.savefig("./topic_dist_train.png")
