# These are the functions that I modified in order to create the synthetic dataset
# for classification. They have to be added to the data_utils.py file.

def create_mixed_dataset(
        stock_model_name="BlackScholes", 
        hyperparam_dict=hyperparam_default,
        vol1 = 0.6,
        seed=0):
    """
    create a synthetic dataset using one of the stock-models. The dataset 
    consists of trajectories of same model but with two different volatility 
    coefficients (all other hyperparameter remain the same). The Dataset is balanced,
    i.e. there are half of the trajectories with volatility vol1 and half with
    hyperparam_dict['volatility']  
    :param stock_model_name: str, name of the stockmodel, see _STOCK_MODELS
    :param hyperparam_dict: dict, contains all needed parameters for the model 
    (contains the value of the first volatility coefficient)
    :param vol1: value of the second volatility coefficient 
    :param seed: int, random seed for the generation of the dataset
    :return: str (path where the dataset is saved), int (time_id to identify
                the dataset)
    """
    df_overview, data_overview = get_dataset_overview()

    np.random.seed(seed=seed)
    hyperparam_dict['model_name'] = stock_model_name
    obs_perc = hyperparam_dict['obs_perc']
    
    #generate the trajectories with the first volatility coefficient
    stockmodel = _STOCK_MODELS[stock_model_name](**hyperparam_dict)
    stock_paths0, dt = stockmodel.generate_paths()
    
    #generate the trajectories with volatility coefficient vol1
    hyperparam_dict['volatility'] = vol1
    stockmodel1 = _STOCK_MODELS[stock_model_name](**hyperparam_dict)
    stock_paths1, dt = stockmodel1.generate_paths()
    #dt is the same as we keep all the parameter but volatility fixed
    
    stock_paths = np.concatenate((stock_paths0,stock_paths1),axis = 0)

    #add label for the trajectories with the different coeffs
    labels = np.concatenate((np.zeros((hyperparam_dict['nb_paths'])),np.ones((hyperparam_dict['nb_paths']))),axis = 0)
    
    size = stock_paths.shape
    observed_dates = np.random.random(size=(size[0], size[2]))
    observed_dates = (observed_dates < obs_perc)*1
    nb_obs = np.sum(observed_dates[:, 1:], axis=1)
    time_id = int(time.time())
    file_name = '{}-{}-{}'.format(stock_model_name,'mixed', time_id)
    path = '{}{}/'.format(training_data_path, file_name)
    desc = json.dumps(hyperparam_dict, sort_keys=True)
    if os.path.exists(path):
        print('Path already exists - abort')
        raise ValueError
    df_app = pd.DataFrame(
        data=[[stock_model_name, time_id, desc]],
        columns=['name', 'id', 'description']
    )
    df_overview = pd.concat([df_overview, df_app],
                            ignore_index=True)
    df_overview.to_csv(data_overview)

    hyperparam_dict['dt'] = dt
    os.makedirs(path)
    with open('{}data.npy'.format(path), 'wb') as f:
        np.save(f, stock_paths)
        np.save(f, observed_dates)
        np.save(f, nb_obs)
        np.save(f, labels)
    with open('{}metadata.txt'.format(path), 'w') as f:
        json.dump(hyperparam_dict, f, sort_keys=True)

    # stock_path dimension: [2*nb_paths, dimension, time_steps]
    return path, time_id

def load_dataset_classification(stock_model_name="BlackScholes", time_id=None):
    """
    load a saved dataset by its name and id
    :param stock_model_name: str, name
    :param time_id: int, id
    :return: np.arrays of stock_paths, observed_dates, number_observations
                dict of hyperparams of the dataset
    """
    time_id = _get_time_id(stock_model_name=stock_model_name, time_id=time_id)
    path = '{}{}-{}/'.format(training_data_path, stock_model_name, int(time_id))

    with open('{}data.npy'.format(path), 'rb') as f:
        stock_paths = np.load(f)
        observed_dates = np.load(f)
        nb_obs = np.load(f)
        labels = np.load(f)
    with open('{}metadata.txt'.format(path), 'r') as f:
        hyperparam_dict = json.load(f)

    return stock_paths, observed_dates, nb_obs, labels, hyperparam_dict
    
class IrregularMixedDataset(Dataset):
    """
    class for iterating over a mixed dataset 
    """
    def __init__(self, model_name, time_id=None, idx=None):
        stock_paths, observed_dates, nb_obs, labels, hyperparam_dict = load_dataset_classification(
            stock_model_name=model_name, time_id=time_id)
        #print(labels)
        if idx is None:
            idx = np.arange(2*hyperparam_dict['nb_paths'])
        self.metadata = hyperparam_dict
        self.stock_paths = stock_paths[idx]
        self.observed_dates = observed_dates[idx]
        self.nb_obs = nb_obs[idx]
        self.labels = labels[idx]
    def __len__(self):
        return len(self.nb_obs)

    def __getitem__(self, idx):
        if type(idx) == int:
            idx = [idx]
        # stock_path dimension: [BATCH_SIZE, DIMENSION, TIME_STEPS]
        return {"idx": idx, "stock_path": self.stock_paths[idx], 
                "observed_dates": self.observed_dates[idx], 
                "nb_obs": self.nb_obs[idx], 'labels': self.labels[idx], "dt": self.metadata['dt']}

def custom_collate_fn_classification(batch):
    """
    function used in torch.DataLoader to construct the custom training batch out
    of the batch of data (non-standard transformations can be applied here)
    :param batch: the input batch (as returned by IrregularDataset)
    :return: a batch (as dict) as needed for the training in train.train
    """
    dt = batch[0]['dt']
    stock_paths = np.concatenate([b['stock_path'] for b in batch], axis=0)
    observed_dates = np.concatenate([b['observed_dates'] for b in batch],
                                    axis=0)
    nb_obs = torch.tensor(np.concatenate([b['nb_obs'] for b in batch], axis=0))
    
    labels = np.concatenate([b['labels'] for b in batch],axis = 0)
    
    start_X = torch.tensor(stock_paths[:,:,0], dtype=torch.float32)
    X = []
    times = []
    time_ptr = [0]
    obs_idx = []
    current_time = 0.
    counter = 0
    for t in range(1, observed_dates.shape[1]):
        current_time += dt
        if observed_dates[:, t].sum() > 0:
            times.append(current_time)
            for i in range(observed_dates.shape[0]):
                if observed_dates[i, t] == 1:
                    counter += 1
                    X.append(stock_paths[i, :, t])
                    obs_idx.append(i)
            time_ptr.append(counter)

    assert len(obs_idx) == observed_dates[:,1:].sum()

    res = {'times': np.array(times), 'time_ptr': np.array(time_ptr),
           'obs_idx': torch.tensor(obs_idx, dtype=torch.long),
           'start_X': start_X, 'n_obs_ot': nb_obs, 
           'X': torch.tensor(X, dtype=torch.float32),
           'true_paths': stock_paths, 'observed_dates': observed_dates,'labels': labels}
    return res