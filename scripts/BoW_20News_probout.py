import sys, os
import numpy as np
from scipy.sparse import csr_matrix
# import theano
# import theano.tensor as T
# from keras.layers import Input, Dense
# from keras.models import Model
from scipy.interpolate import griddata
import multiprocessing as mp


def GetRankingOrder(query, database):
    scores = np.dot(database, query.reshape(-1, 1)).reshape(-1)
    return np.argsort(-scores)


def Normalize(data):
    return data / np.sqrt((data ** 2).sum(axis=1)).reshape(-1, 1)


def GetRelevanceLabels(label, database_labels, multilabel=False):
    if multilabel:
        r = database_labels[:, label].reshape(-1)
    else:
        r = 1.0 * (label == database_labels)
    return r


def GetPrecisionRecall(rank_order, label, database_labels, multilabel=False):
    db_size = database_labels.shape[0]
    relevance_labels = GetRelevanceLabels(label, database_labels, multilabel=multilabel)
    relevance_labels = relevance_labels[rank_order]
    total_positives = float(relevance_labels.sum())
    total_positives += 1e-10
    cumsum = np.cumsum(relevance_labels)
    precision = cumsum / (1.0 + np.arange(db_size))
    recall = cumsum / total_positives
    ap = (precision * relevance_labels).sum() / total_positives
    return precision, recall, ap


def LoadFiles(files):
    data = None
    for f in files:
        if data is None:
            data = np.load(f)
        else:
            data = np.concatenate((data, np.load(f)))
    return data


def ApplyGrid2(r, p, grid_recall):
    return griddata(r, p, grid_recall, 'nearest', fill_value=1.0).reshape(-1)


def ApplyGrid(r, p, g):
    output = np.zeros(g.shape)
    index = 0
    while g[index] <= r[0]:
        output[index] = p[0]
        index += 1
    break_out = False
    num_grid_points = g.shape[0]
    for i in range(1, r.shape[0]):
        while g[index] > r[i - 1] and g[index] <= r[i]:
            output[index] = p[i - 1]
            index += 1
            if index == num_grid_points:
                break_out = True
                break
        if break_out:
            break
    return output


def f(start, end, queries_rep, database_rep, queries_truth, database_truth,
      multilabel, grid_recall, numlabels, parallel):
    precision_matrix = np.zeros((numlabels, grid_recall.shape[0]))
    ap_matrix = np.zeros((numlabels))
    per_query_ap_matrix = np.zeros((end - start))
    per_query_prec = np.zeros((end - start))
    count = np.zeros((numlabels))
    db_size = database_truth.shape[0]
    for i in range(start, end):
        query = queries_rep[i]
        #     if start == 0:
        #       sys.stdout.write('\r%d' % (i+1))
        #       sys.stdout.flush()
        rank_order = GetRankingOrder(query, database_rep)
        labels = GetLabels(queries_truth[i], multilabel=multilabel)
        for label in labels:
            precision, recall, ap = GetPrecisionRecall(rank_order, label, database_truth, multilabel=multilabel)
            if parallel:
                precision_matrix[label, :] += ApplyGrid(recall, precision, grid_recall).reshape(-1)
            else:
                precision_matrix[label, :] += ApplyGrid2(recall, precision, grid_recall).reshape(-1)
            ap_matrix[label] += ap
            count[label] += 1
            per_query_ap_matrix[i - start] += ap
            per_query_prec[i - start] += precision[min(50, db_size - 1)]
        per_query_ap_matrix[i - start] /= len(labels)
        per_query_prec[i - start] /= len(labels)

    return precision_matrix, ap_matrix, count, per_query_ap_matrix, per_query_prec


def GetPrecisionMatrix(database_rep, database_truth,
                       queries_rep, queries_truth,
                       numqueries='all', numdatabase='all', grid_size=0.001,
                       parallelize=False, multilabel=False):
    #   database_rep = LoadFiles(database_rep_files)
    #   queries_rep = LoadFiles(queries_rep_files)
    if numdatabase == 'all':
        numdatabase = database_rep.shape[0]
    if numqueries == 'all':
        numqueries = queries_rep.shape[0]
    database_rep = database_rep[:numdatabase]
    queries_rep = queries_rep[:numqueries]
    #   _, ext = os.path.splitext(database_truth_file)
    #   if ext == '.npz':
    #     database_truth = LoadSparse(database_truth_file)[:numdatabase].toarray()
    #     multilabel = True
    #   else:
    #     database_truth = np.load(database_truth_file)[:numdatabase]
    #     multilabel = False
    #
    #   _, ext = os.path.splitext(queries_truth_file)
    #   if ext == '.npz':
    #     queries_truth = LoadSparse(queries_truth_file)[:numqueries].toarray()
    #   else:
    #     queries_truth = np.load(queries_truth_file)[:numqueries]


    assert database_rep.shape[0] == database_truth.shape[0], \
        'Database has shape %s but truth has shape %s' % (
            database_rep.shape, database_truth.shape)
    assert queries_rep.shape[0] == queries_truth.shape[0], \
        'Queries have shape %s but truth has shape %s' % (
            queries_rep.shape, queries_truth.shape)

    if multilabel:
        numlabels = queries_truth.shape[1]
    else:
        numlabels = 20
    # Normalize ?
    queries_rep = Normalize(queries_rep)
    database_rep = Normalize(database_rep)

    # Compute precison and recall.
    grid_recall = np.concatenate((
        np.arange(0.0, 0.01, 0.00001),
        np.arange(0.01, 1.0, grid_size),
    ))

    precision_matrix = np.zeros((numlabels, grid_recall.shape[0]))
    ap_matrix = np.zeros((numlabels))
    count = np.zeros((numlabels))
    num_queries = queries_rep.shape[0]
    pq_ap = np.zeros((num_queries))
    pq_prec = np.zeros((num_queries))
    results_list = []
    if parallelize:
        num_processes = min(num_queries, mp.cpu_count())
        pool = mp.Pool(num_processes)
        num_queries_per_process = num_queries / num_processes
        left_overs = num_queries % num_processes
        results = []
        end = 0
        i = 0
        while end < num_queries:
            i += 1
            start = end
            end = min(start + num_queries_per_process, num_queries)
            if left_overs > 0:
                end += 1
                left_overs -= 1
            print 'Process %d: %d - %d ' % (i, start, end)
            results.append(pool.apply_async(f, (start, end, queries_rep, database_rep,
                                                queries_truth, database_truth,
                                                multilabel, grid_recall, numlabels, parallelize)))
        print 'Launched %d processes.' % num_processes
        q_index = 0
        l = []
        for res in results:
            l.append(res.get())
        for p, a, c, pq, pq_p in [res.get() for res in results]:
            precision_matrix += p
            ap_matrix += a
            count += c
            num_q = pq.shape[0]
            pq_ap[q_index:q_index + num_q] = pq
            pq_prec[q_index:q_index + num_q] = pq_p
            q_index += num_q
    else:
        start = 0
        end = num_queries
        precision_matrix, ap_matrix, count, pq_ap, pq_prec = f(start, end, queries_rep,
                                                               database_rep, queries_truth,
                                                               database_truth, multilabel,
                                                               grid_recall, numlabels, parallelize)
    sys.stdout.write('\n')
    nnz = (count > 0).sum()
    count += 1e-10
    precision_matrix /= count.reshape(-1, 1)
    ap_matrix /= count
    MAP = ap_matrix.sum() / nnz
    # print 'MAP %.4f' % MAP
    # print 'approx MAP before 0.2: %.4f ' % np.mean(precision_matrix[:, :int((0.2-grid_recall>0).sum())])
    return precision_matrix, grid_recall, ap_matrix, pq_ap, pq_prec


def GetLabels(query, multilabel=False):
    if multilabel:
        return query.nonzero()[0].tolist()
    else:
        return [query]


def LoadSparse(inputfile, verbose=False):
    """Loads a sparse matrix stored as npz file."""
    npzfile = np.load(inputfile)
    mat = csr_matrix((npzfile['data'], npzfile['indices'],
                      npzfile['indptr']),
                     shape=tuple(list(npzfile['shape'])))
    if verbose:
        print 'Loaded sparse matrix from %s of shape %s' % (inputfile,
                                                            mat.shape.__str__())
    return mat


def SaveSparse(outputfile, mat, verbose=False):
    if verbose:
        print 'Saving to %s shape %s' % (outputfile, mat.shape.__str__())
    np.savez(outputfile, data=mat.data, indices=mat.indices, indptr=mat.indptr,
             shape=np.array(list(mat.shape)))


epsilon = 1.0e-9


def AE_nll(count_each_word, prob_each_word):
    '''
    
    :param count_each_word: batch_size x voc_size, the count for each word 
    :param prob_each_word: batch_size x voc_size, softmax output for each word
    :return: 
    '''
    y_pred = T.clip(prob_each_word, epsilon, 1.0 - epsilon)
    y_pred /= y_pred.sum(axis=-1, keepdims=True)
    nll = -T.sum(count_each_word * T.log(y_pred), axis=y_pred.ndim - 1)
    return nll


# sys.argv.pop(0)

# dataset_path = sys.argv[0]
# encoding_dim = int(sys.argv[1])
# n_epoch = int(sys.argv[2])

# trainset_hist = LoadSparse(os.path.join(dataset_path, 'train_data.npz')).toarray()
# validset_hist = LoadSparse(os.path.join(dataset_path, 'valid_data.npz')).toarray()
# testset_hist = LoadSparse(os.path.join(dataset_path, 'test_data.npz')).toarray()
# trainset_label_raw = np.squeeze(np.load(os.path.join(dataset_path, 'train_labels.npy')))
# validset_label_raw = np.squeeze(np.load(os.path.join(dataset_path, 'valid_labels.npy')))
# testset_label_raw = np.squeeze(np.load(os.path.join(dataset_path, 'test_labels.npy')))

# trainset_hist = trainset_hist.astype(theano.config.floatX)
# validset_hist = validset_hist.astype(theano.config.floatX)
# testset_hist = testset_hist.astype(theano.config.floatX)

# input_dim = trainset_hist.shape[1]
# input_hist = Input(shape=(input_dim,))
# encoded = Dense(encoding_dim, activation='tanh')(input_hist)
# decoded = Dense(input_dim, activation='softmax')(encoded)
# autoencoder = Model(input_hist, decoded)
# encoder = Model(input_hist, encoded)
# encoded_input = Input(shape=(encoding_dim,))
# decoder_layer = autoencoder.layers[-1]
# decoder = Model(encoded_input, decoder_layer(encoded_input))

# autoencoder.compile(optimizer='adam', loss=AE_nll)
# autoencoder.fit(trainset_hist, trainset_hist,
#                 batch_size=512,
#                 epochs=n_epoch,
#                 shuffle=True,
#                 validation_data=(validset_hist, validset_hist))

# rep_train = encoder.predict(trainset_hist)
# rep_valid = encoder.predict(validset_hist)
# rep_test = encoder.predict(testset_hist)
# rep_trainval = np.vstack((rep_train, rep_valid))
# label_trainval = np.concatenate((trainset_label_raw, validset_label_raw), axis=0)
# precision, grid_recall, ap, pq_ap, pq_prec = GetPrecisionMatrix(
#     rep_trainval, label_trainval, rep_test, testset_label_raw,
#     numqueries='all', numdatabase='all', parallelize=False, multilabel=False)

# mAP_final = np.mean(ap)

# prfile = '20News_%s.npz' % encoding_dim
# print 'Writing precision, recall matrices to %s' % prfile
# # np.savez(prfile, prec=precision, grid_recall=grid_recall, ap=ap, pq_ap=pq_ap, pq_prec=pq_prec)
# np.savez(prfile, precision=precision, recall=grid_recall)
