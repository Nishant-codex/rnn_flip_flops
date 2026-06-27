
 


#Directory containing all the trained rate RNN model .mat files
model_dir = '../models/go-nogo/P_rec_0.2_Taus_4.0_20.0';
mat_files = dir(fullfile(model_dir, '*.mat'));

# % Whether to use the initial random connectivity weights
# % This should be set to false unless you want to compare
# % the effects of pre-trained vs post-trained weights
use_initial_weights = False

# % Number of trials to use to evaluate the LIF RNN
n_trials = 100

# % Scaling factor values to try for grid search
# % The more values it has, the longer the search
scaling_factors = np.arange(20,75,5)#[20:5:75];

# % Grid search
# for i length(mat_files)
  curr_fname = a#mat_files(i).name;
  curr_full = b#fullfile(mat_files(i).folder, curr_fname);
  print(['Analyzing ' curr_fname])

  # % Get the task name
  # if ~isempty(findstr(curr_full, 'go-nogo'))
  task_name = 'go-nogo'
  # elseif ~isempty(findstr(curr_full, 'mante'))
  #   task_name = 'mante';
  # elseif ~isempty(findstr(curr_full, 'xor'))
  #   task_name = 'xor';
  # end

  # % Load the model
  load(curr_full);

  # % Skip if the file was run before
  # if exist('opt_scaling_factor'):
  #   clearvars -except model_dir mat_files n_trials scaling_factors use_initial_weights
  #   continue;
  # else
  opt_scaling_factor = np.;
    save(curr_full, 'opt_scaling_factor', '-append');
  end

  # % Go-NoGo task
  if task_name == 'go-nogo':
    down_sample = 1
    all_perfs = zeros(length(scaling_factors), 1);

    for k in range(len(scaling_factors)):
      outs = np.zeros(n_trials, 20000)
      trials = np.zeros(n_trials, 1)
      perfs = np.zeros(n_trials, 1)

      scaling_factor = scaling_factors[k]
      print(scaling_factor)

      for j in range(n_trials):
        u = np.zeros(1, 201)
        if rand >= 0.50
          u(51:75) = 1.0;
          trials(j) = 1;
      
        stims = struct();
        stims.mode = 'none';
        [W, REC, spk, rs, all_fr, out, params] = LIF_network_fnc(curr_full, scaling_factor,...
        u, stims, down_sample, use_initial_weights);
        outs[j, :] = out
        if np.max(out[10000:]) > 0.7 & trials[j] == 1:
          perfs[j] = 1
        elif np.max(out[10000:]) < 0.3 & trials[j] == 0:
          perfs[j] = 1
      all_perfs[k] = mean(perfs);

    [v, ind] = np.max(all_perfs)
    [v, scaling_factors(ind)]

    # % Save the optimal scaling factor
    opt_scaling_factor = scaling_factors(ind);
    save(curr_full, 'opt_scaling_factor', 'all_perfs', 'scaling_factors', '-append');
    clear opt_scaling_factor;

