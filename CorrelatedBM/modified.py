def create_correlated_dataset(
        stock_model_name="BlackScholes",
        hyperparam_dict=hyperparam_default,
        rho = 0.6, vol1 = 0.6,
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
    
    

    hyperparam_dict['nb_paths'] = int(hyperparam_dict['nb_paths']/2)
    #generate the trajectories with the first volatility coefficient
    stockmodel = _STOCK_MODELS[stock_model_name](**hyperparam_dict)
    stock_paths0, dt = stockmodel.generate_correlated_paths(rho = rho)



    #generate the trajectories with volatility coefficient vol1
    hyperparam_dict['volatility'] = vol1
    stockmodel1 = _STOCK_MODELS[stock_model_name](**hyperparam_dict)
    stock_paths1, dt = stockmodel1.generate_correlated_paths(rho = rho)
    #dt is the same as we keep all the parameter but volatility fixed

    stock_paths = np.concatenate((stock_paths0,stock_paths1),axis = 0)

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

    # stock_path dimension: [nb_paths, dimension, time_steps]
    return path, time_id
    
def generate_correlated_paths(self, start_X=None, rho=0.6):
      """
      Generates correlated paths.
      Arguments:
          start_X (see generate_paths)
          rho (float): float between 0. and 1.
      Return:
          spot_paths (np.array): paths generated by the Black-Scholes model.
          spot_paths2 (np.array): paths generated by the Black-Scholes model
              and correlated to spot_paths.
          dt (array like): time range
      """
      ## changed to make it generate 2 dim trajectories with corr coordinates
      #instead of 1 dim trajectories paarwise correlated
      if (rho > 1.) or (rho < 0.):
          raise ValueError("rho should be 0. <= rho <= 1.")

      drift = lambda x, t: self.drift*self.periodic_coeff(t)*x
      diffusion = lambda x, t: self.volatility * x
      spot_paths = np.empty(
          (self.nb_paths, self.dimensions, self.nb_steps + 1))
      dt = self.maturity / self.nb_steps
      if start_X is not None:
          spot_paths[:, :, 0] = start_X #has to be of dimension self.dim
      for i in range(self.nb_paths):
          if start_X is None:
              spot_paths[i, :, 0] = self.S0 #the initial value S0 has dim = self.dimension
          for k in range(1, self.nb_steps + 1):
              random_numbers = np.random.normal(0, 1, self.dimensions)

              dW = random_numbers[0] * np.sqrt(dt)
              dW2 = (random_numbers[0] * rho + np.sqrt(1 - rho**2) * random_numbers[1]) * np.sqrt(dt)
              spot_paths[i, 0, k] = (
                      spot_paths[i, 0, k - 1]
                      + drift(spot_paths[i, 0, k - 1], (k-1) * dt) * dt
                      + np.multiply(diffusion(spot_paths[i, 0, k - 1], (k) * dt),
                               dW))
              
              spot_paths[i, 1, k] = (
                      spot_paths[i, 1, k - 1]
                      + drift(spot_paths[i, 1, k - 1], (k-1) * dt) * dt
                      + np.multiply(diffusion(spot_paths[i, 1, k - 1], (k) * dt),
                               dW2))

      return spot_paths, dt