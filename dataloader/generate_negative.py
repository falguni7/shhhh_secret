import numpy as np
import random
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter
import pickle
import os
import matplotlib.pyplot as plt

def sine(length, freq=0.04, coef=1.5, offset=0.0, noise_amp=0.05):
    # timestamp = np.linspace(0, 10, length)
    timestamp = np.arange(length)
    value = np.sin(2 * np.pi * freq * timestamp)
    if noise_amp != 0:
        noise = np.random.normal(0, 1, length)
        value = value + noise_amp * noise
    value = coef * value + offset
    return value


def square_sine(level=5, length=500, freq=0.04, coef=1.5, offset=0.0, noise_amp=0.05):
    value = np.zeros(length)
    for i in range(level):
        value += 1 / (2 * i + 1) * sine(length=length, freq=freq * (2 * i + 1), coef=coef, offset=offset, noise_amp=noise_amp)
    return value


# Add collective point outliers to original data (Variant)
def point_outliers(train_x, configs):
    for i, x_i in enumerate(train_x):
        if x_i.shape[1] > 1:
            elements = np.arange(0, x_i.shape[1])
            if configs.dim > 1:
                dim_size = np.random.randint(1, configs.dim)
            else:
                dim_size = 1
            selected_elements = np.random.choice(elements, size=dim_size, replace=False)
            for item in selected_elements:
                position = int(np.random.rand() * train_x.shape[1])
                local_std = x_i[:, item].std()
                local_mean = x_i[:, item].mean()
                scale = local_std * np.random.choice((-1, 1)) * 3 * (np.random.rand() + 1)
                point_value = local_mean + scale
                train_x[i, position, item] = point_value
        else:
            position = int(np.random.rand() * train_x.shape[1])
            local_std = x_i[:, 0].std()
            local_mean = x_i[:, 0].mean()
            scale = local_std * np.random.choice((-1, 1)) * 3 * (np.random.rand()+1)
            point_value = local_mean + scale
            train_x[i, position, 0] = point_value
    return train_x


# Add collective trend outliers to original data (Variant)
def collective_trend_outliers(train_x, configs):
    for i, x_i in enumerate(train_x):
        if x_i.shape[1] > 1:
            elements = np.arange(0, x_i.shape[1])
            if configs.dim > 1:
                dim_size = np.random.randint(1, configs.dim)
            else:
                dim_size = 1
            selected_elements = np.random.choice(elements, size=dim_size, replace=False)
            for item in selected_elements:
                radius = max(int(train_x.shape[1]/6), int(np.random.rand() * train_x.shape[1]))
                factor = np.random.rand() * configs.trend_rate
                position = int(np.random.rand() * (train_x.shape[1] - radius))
                start, end = position, position + radius
                slope = np.random.choice([-1, 1]) * factor * np.arange(end - start)
                train_x[i, start:end, item] = x_i[start:end, item] + slope
        else:
            radius = max(int(train_x.shape[1] / 6), int(np.random.rand() * train_x.shape[1]))
            factor = np.random.rand() * configs.trend_rate
            position = int(np.random.rand() * (train_x.shape[1] - radius))
            start, end = position, position + radius
            slope = np.random.choice([-1, 1]) * factor * np.arange(end - start)
            train_x[i, start:end, 0] = x_i[start:end, 0] + slope
    return train_x


# Add collective seasonal outliers to original data
def collective_seasonal_outliers(train_x):
    seasonal_config = {'length': 400, 'freq': 0.04, 'coef': 1.5, "offset": 0.0, 'noise_amp': 0.05}
    for i, x_i in enumerate(train_x):
        radius = max(int(train_x.shape[1]/6), int(np.random.rand() * train_x.shape[1]))
        factor = np.random.rand()
        seasonal_config['freq'] = factor * seasonal_config['freq']
        position = int(np.random.rand() * (train_x.shape[1] - radius))
        start, end = position, position + radius
        train_x[i, start:end, 0] = sine(**seasonal_config)[start:end]
    return train_x


# Add cut outliers to original data (Variant)
def cut_outliers(train_x):
    for i, x_i in enumerate(train_x):
        radius = max(int(train_x.shape[1]/6), int(np.random.rand() * (train_x.shape[1]-2)))
        to_position = int(random.uniform(0, train_x.shape[1] - radius))
        train_x[i, to_position:to_position + radius, :] = 0
    return train_x


# From the same sequence, Add outliers to original data via the CutPaste method (Variant)
def cut_paste_outliers_same(train_x):
    for i, x_i in enumerate(train_x):
        radius = max(int(train_x.shape[1]/6), int(np.random.rand() * (train_x.shape[1]-2)))
        cut_data = x_i
        position = random.sample(range(0, train_x.shape[1] - radius + 1), 2)
        from_position = position[0]
        to_position = position[1]
        cut_data = cut_data[from_position:from_position + radius, :]
        train_x[i, to_position:to_position + radius, :] = cut_data[:, :]
    return train_x


# From different sequences, Add outliers to original data via the CutPaste method (Variant)
def cut_paste_outliers_other(train_x):
    for i, x_i in enumerate(train_x):
        radius = max(int(train_x.shape[1]/6), int(np.random.rand() * (train_x.shape[1]-2)))
        cut_random = int(random.uniform(0, train_x.shape[0]))
        cut_data = train_x[cut_random, :, :]
        from_position = int(random.uniform(0, train_x.shape[1] - radius))
        to_position = int(random.uniform(0, train_x.shape[1] - radius))
        train_x[i, to_position:to_position + radius, :] = cut_data[from_position:from_position + radius, :]
    return train_x


# For multidimensional time series data, add the same trend to each dimension (Variant)
# CutAddPaste with same trends (SameCAP)
def cut_paste_outliers_same_trend(train_x, configs):
    for i, x_i in enumerate(train_x):
        radius = max(int(train_x.shape[1]/3), int(np.random.rand() * train_x.shape[1]))
        cut_random = int(random.uniform(0, train_x.shape[0]))
        cut_data = train_x[cut_random, :, :]
        from_position = int(np.random.rand() * (train_x.shape[1] - radius))
        cut_data = cut_data[from_position:from_position + radius, :]
        factor = np.random.rand() * configs.trend_rate
        slope = np.random.choice([-1, 1]) * factor * np.arange(radius)
        slope = slope.reshape(-1, 1)
        slope = np.tile(slope, configs.input_channels)
        cut_data[:, :] += slope
        to_position = int(np.random.rand() * (train_x.shape[1] - radius))
        train_x[i, to_position:to_position + radius, :] = cut_data[:, :]
    return train_x


# Add outliers to original data via our CutAddPaste method
def cut_add_paste_outlier(train_x, configs):
    for i, x_i in enumerate(train_x): 
        radius = max(int(configs.cut_rate), int(np.random.rand() * train_x.shape[1])) 

        cut_random = int(random.uniform(0, train_x.shape[0])) 
        cut_data = train_x[cut_random, :, :]
        from_position = int(np.random.rand() * (train_x.shape[1] - radius)) 
        cut_data = cut_data[from_position:from_position + radius, :]    
        if cut_data.shape[1] > 1:   
            elements = np.arange(0, cut_data.shape[1])  
            if configs.dim > 1: 
                dim_size = np.random.randint(1, configs.dim)    
            else:
                dim_size = 1
            selected_elements = np.random.choice(elements, size=dim_size, replace=False)   
            for item in selected_elements:
                factor = np.random.rand() * configs.trend_rate  
                slope = np.random.choice([-1, 1]) * factor * np.arange(radius)  
                cut_data[:, item] += slope
        else:
            factor = np.random.rand() * configs.trend_rate
            slope = np.random.choice([-1, 1]) * factor * np.arange(radius)
            cut_data[:, 0] += slope
        to_position = int(np.random.rand() * (train_x.shape[1] - radius))
        train_x[i, to_position:to_position + radius, :] = cut_data[:, :]
        if i < 10:
            print(f"Sample {i}: radius={radius}, cut_random={cut_random}, from_position={from_position}, to_position={to_position}")
    return train_x


def cut_add_paste_impute_outlier(train_x, configs):
    """BPA Synthetic anomaly generation.
    """
    methods = ['linear', 'pchip', 'savgol']
    n_timesteps = train_x.shape[1]
    n_feats = train_x.shape[2]
    B = train_x.shape[0]
    cap_train_x = np.zeros((B, n_timesteps, n_feats), dtype=train_x.dtype)
    cap_mask = np.zeros((B, n_timesteps, n_feats), dtype=np.uint8)
    out_dir = getattr(configs, 'cap_out_dir', 'results')
    os.makedirs(out_dir, exist_ok=True)

    for i, x_i in enumerate(train_x):
        radius = max(int(configs.cut_rate), int(np.random.rand() * train_x.shape[1])) 

        cut_random = int(random.uniform(0, train_x.shape[0]))
        cut_data = train_x[cut_random, :, :]
        from_position = int(np.random.rand() * (train_x.shape[1] - radius))
        cut_data = cut_data[from_position:from_position + radius, :]

        # optionally add trends to cut
        if cut_data.shape[1] > 1:
            elements = np.arange(0, cut_data.shape[1])
            if configs.dim > 1:
                dim_size = np.random.randint(1, configs.dim)
            else:
                dim_size = 1
            selected_elements = np.random.choice(elements, size=dim_size, replace=False)
            for item in selected_elements:
                factor = np.random.rand() * configs.trend_rate
                slope = np.random.choice([-1, 1]) * factor * np.arange(radius)
                cut_data[:, item] += slope
        else:
            factor = np.random.rand() * configs.trend_rate
            slope = np.random.choice([-1, 1]) * factor * np.arange(radius)
            cut_data[:, 0] += slope

        # pick target position
        to_position = int(np.random.rand() * (train_x.shape[1] - radius + 1))
        to_position = max(0, min(to_position, train_x.shape[1] - radius))

        # paste the cut data
        train_x[i, to_position:to_position + radius, :] = cut_data[:, :]

        # random pre/post imputation lengths
        pre_len = random.choice([3, 4]) 
        post_len = random.choice([3, 4])

        pre_positions = [p for p in range(to_position - pre_len, to_position) if p >= 0]
        post_positions = [p for p in range(to_position + radius, to_position + radius + post_len) if p < n_timesteps]

        cap_train_x[i, :, :] = train_x[i, :, :].copy()
        if len(pre_positions) > 0 or len(post_positions) > 0:
            mask_positions = pre_positions + post_positions
            cap_mask[i, mask_positions, :] = 1

        method = random.choice(methods)
        method_pre = method_post = method
        if random.random() < 0.25:
            method_pre = random.choice(methods)
        if random.random() < 0.25:
            method_post = random.choice(methods)

        for f in range(n_feats):            
            arr = train_x[i, :, f].astype(float).copy()
            src_idx = np.arange(to_position, to_position + radius, dtype=float)
            src_y = arr[src_idx.astype(int)]
            outside_idx = [idx for idx in range(n_timesteps) if idx not in (pre_positions + post_positions)]
            known_x = np.array(outside_idx, dtype=float)
            known_y = arr[known_x.astype(int)]

            if known_x.size < 2:
                for pos in pre_positions + post_positions:
                    if pos == 0:
                        arr[pos] = arr[pos + 1]
                    elif pos == n_timesteps - 1:
                        arr[pos] = arr[pos - 1]
                    else:
                        arr[pos] = 0.5 * (arr[pos - 1] + arr[pos + 1])
                train_x[i, :, f] = arr
                continue

            include_edge = min(3, radius)

            # Left-edge augmented known set (for pre interpolation)
            left_edge_idx = src_idx[:include_edge]
            left_known_x = np.concatenate((known_x[known_x < left_edge_idx[0]], left_edge_idx)) if known_x.size > 0 else left_edge_idx
            left_known_y = np.concatenate((known_y[known_x < left_edge_idx[0]], src_y[:include_edge])) if known_x.size > 0 else src_y[:include_edge]

            # Right-edge augmented known set (for post interpolation)
            right_edge_idx = src_idx[-include_edge:]
            right_known_x = np.concatenate((right_edge_idx, known_x[known_x > right_edge_idx[-1]])) if known_x.size > 0 else right_edge_idx
            right_known_y = np.concatenate((src_y[-include_edge:], known_y[known_x > right_edge_idx[-1]])) if known_x.size > 0 else src_y[-include_edge:]

            # Interpolate pre_positions using left_known_* sets
            y_pre = np.array([])
            if len(pre_positions) > 0:
                x_fill_pre = np.array(pre_positions, dtype=float)
                try:
                    if method_pre == 'linear':
                        y_pre = np.interp(x_fill_pre, left_known_x, left_known_y)
                    elif method_pre == 'pchip':
                        interp = PchipInterpolator(left_known_x, left_known_y, extrapolate=True)
                        y_pre = interp(x_fill_pre)
                    elif method_pre == 'savgol':
                        tmp = arr.copy()
                        missing = np.array(pre_positions + post_positions, dtype=int)
                        tmp[missing] = np.interp(missing.astype(float), known_x, known_y)
                        win = min(9, n_timesteps if n_timesteps % 2 == 1 else n_timesteps - 1)
                        if win < 3:
                            y_pre = tmp[x_fill_pre.astype(int)]
                        else:
                            y_pre = savgol_filter(tmp, window_length=win, polyorder=2, mode='nearest')[x_fill_pre.astype(int)]
                    else:
                        y_pre = np.interp(x_fill_pre, left_known_x, left_known_y)
                except Exception:
                    print(f"Warning: Exception occurred during pre interpolation for sample {i}, feature {f}. Falling back to linear interpolation.")
                    y_pre = np.interp(x_fill_pre, left_known_x, left_known_y)

            # Interpolate post_positions using right_known_* sets
            y_post = np.array([])
            if len(post_positions) > 0:
                x_fill_post = np.array(post_positions, dtype=float)
                try:
                    if method_post == 'linear':
                        y_post = np.interp(x_fill_post, right_known_x, right_known_y)
                    elif method_post == 'pchip':
                        interp = PchipInterpolator(right_known_x, right_known_y, extrapolate=True)
                        y_post = interp(x_fill_post)
                    elif method_post == 'savgol':
                        tmp = arr.copy()
                        missing = np.array(pre_positions + post_positions, dtype=int)
                        tmp[missing] = np.interp(missing.astype(float), known_x, known_y)
                        win = min(9, n_timesteps if n_timesteps % 2 == 1 else n_timesteps - 1)
                        if win < 3:
                            y_post = tmp[x_fill_post.astype(int)]
                        else:
                            y_post = savgol_filter(tmp, window_length=win, polyorder=2, mode='nearest')[x_fill_post.astype(int)]
                    else:
                        y_post = np.interp(x_fill_post, right_known_x, right_known_y)
                except Exception:
                    print(f"Warning: Exception occurred during post interpolation for sample {i}, feature {f}. Falling back to linear interpolation.")
                    y_post = np.interp(x_fill_post, right_known_x, right_known_y)

            # apply small jitter
            base_noise = getattr(configs, 'impute_noise', 0.01)
            if y_pre.size > 0:
                y_pre = y_pre + np.random.normal(0, base_noise * random.uniform(0.5, 1.5), y_pre.shape)
            if y_post.size > 0:
                y_post = y_post + np.random.normal(0, base_noise * random.uniform(0.5, 1.5), y_post.shape)

            # assign the interpolated values to outside positions
            for k, pos in enumerate(pre_positions):
                arr[pos] = y_pre[k]
            for k, pos in enumerate(post_positions):
                arr[pos] = y_post[k]

            if pre_len > 0 and to_position - 1 >= 0:
                left_val = arr[to_position - 1]
                for b in range(pre_len):
                    idx = to_position + b
                    if idx < to_position + radius:
                        alpha = random.uniform(0.3, 0.9)
                        arr[idx] = alpha * arr[idx] + (1 - alpha) * left_val

            if post_len > 0 and to_position + radius < n_timesteps:
                right_val = arr[to_position + radius]
                for b in range(post_len):
                    idx = to_position + radius - 1 - b
                    if idx >= to_position:
                        alpha = random.uniform(0.3, 0.9)
                        arr[idx] = alpha * arr[idx] + (1 - alpha) * right_val

            # write back
            train_x[i, :, f] = arr

        if i < 10:
            print(f"Sample {i}: chosen_radius={radius}, cut_random={cut_random}, from_position={from_position}, to_position={to_position}, pre_len={pre_len}, post_len={post_len}")

    try:
        with open(os.path.join(out_dir, 'cap_train_x.pkl'), 'wb') as f:
            pickle.dump(cap_train_x, f)
        with open(os.path.join(out_dir, 'cap_mask.pkl'), 'wb') as f:
            pickle.dump(cap_mask, f)
        with open(os.path.join(out_dir, 'capi_train_x.pkl'), 'wb') as f:
            pickle.dump(train_x, f)
    except Exception as e:
        print(f"Warning: failed to save cap files: {e}")

    print("Cut-Add-Paste-Impute augmentation completed.")
    return train_x