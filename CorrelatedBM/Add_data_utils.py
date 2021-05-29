"""
Methods to add to data_utils.py
"""

class IrregularCorrelatedDataset(Dataset):
    """
    class for iterating over a correlated dataset
    """
    def __init__(self, model_name, time_id=None, idx=None):
        stock_paths, observed_dates, nb_obs, labels, hyperparam_dict = load_dataset_classification(
            stock_model_name=model_name, time_id=time_id)
        if idx is None:
            idx = np.arange(hyperparam_dict['nb_paths'])

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


def create_correlated_dataset(
        stock_model_name="BlackScholes",
        hyperparam_dict=hyperparam_default,
        rho = 0.6,
        seed=0):
    """
    create a synthetic dataset using one of the stock-models. The dataset
    consists of trajectories of same model but correlated, with a correlation
    coefficient rho. The Dataset is balanced, i.e. there are half of the
    trajectories with volatility vol1 and half withhyperparam_dict['volatility']
    :param stock_model_name: str, name of the stockmodel, see _STOCK_MODELS
    :param hyperparam_dict: dict, contains all needed parameters for the model
    (contains the value of the first volatility coefficient)
    :param rho: correlation coefficient
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
    stock_paths0, stock_paths1, dt = stockmodel.generate_correlated_paths(rho=rho)
    stock_paths = np.concatenate((stock_paths0, stock_paths1),axis = 0)

    #add label for the trajectories with the different coeffs
    labels = np.concatenate((np.zeros((hyperparam_dict['nb_paths'])),np.ones((hyperparam_dict['nb_paths']))),axis = 0)

    hyperparam_dict['nb_paths'] = int(2*hyperparam_dict['nb_paths'])
    size = stock_paths.shape
    observed_dates = np.random.random(size=(size[0], size[2]))
    observed_dates = (observed_dates < obs_perc)*1
    nb_obs = np.sum(observed_dates[:, 1:], axis=1)
    time_id = int(time.time())
    file_name = '{}-{}-{}'.format(stock_model_name,'corr', time_id)
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
