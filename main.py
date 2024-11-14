import torch
import numpy as np
import os
import yaml
import gc
import re


class TensorCreator:

    def __init__(self, config: str = "config.yml"):
        self.timespan = None
        self.path = None
        self.savefile_prefix = None
        self.divide_by_channels = None
        self.file_type = None
        self.quaternion_file_type = None
        self.instruction = None
        self.translate_hex = None

        try:
            with open(config, 'r') as config_file:
                self.cfg = yaml.load(config_file, Loader=yaml.FullLoader)
        except FileNotFoundError as ex:
            print(f"Error: There is no such file as {config}.\n Exception: {ex}")

    def set_creation_params(self):
        self.instruction = self.cfg["TensorCreator"]["CreatorParams"]["instruction"]
        self.quaternion_file_type = self.cfg["TensorCreator"]["CreatorParams"]["quaternion_file_type"]
        self.file_type = self.cfg["TensorCreator"]["CreatorParams"]["file_type"]
        self.divide_by_channels = self.cfg["TensorCreator"]["CreatorParams"]["divide_by_channels"]
        self.timespan = self.cfg["TensorCreator"]["CreatorParams"]["timespan"]
        self.translate_hex = self.cfg["TensorCreator"]["CreatorParams"]["translate_hex"]
        self.savefile_prefix = self.cfg["TensorCreator"]["FileParams"]["save_file_prefix_path"]
        self.path = self.cfg["TensorCreator"]["FileParams"]["path_to_main_dir"]

    def init_tensor_creation_process(self):
        if self.timespan == "Every half year":
            self.init_half_year_tensors()
        elif self.timespan == "Every year":
            self.init_year_tensors()
        elif self.timespan == "All at once":
            self.init_alldata_tensors()
        elif self.timespan == "By Channels":
            self.init_channel_tensors()
        else:
            print("The process is not supported.")

    def init_half_year_tensors(self):
        half_year_dirs = [d for d in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, d))]
        total_dirs = len(half_year_dirs)

        for scanned_dirs, half_year_dir in enumerate(half_year_dirs, start=1):
            subdir_path = os.path.join(self.path, half_year_dir)
            data_list = []

            print(f"Progress: {(scanned_dirs / total_dirs) * 100:.2f}% directories scanned.")

            for root, _, files in os.walk(subdir_path):
                if any(file.endswith(self.quaternion_file_type) for file in files):
                    data_list += self._process_files(files, root)
            self._save_combined_tensor(data_list, f"{self.savefile_prefix}_half_year_{half_year_dir}.pt")
            del data_list

    def init_year_tensors(self):
        year_dirs = sorted({d[:4] for d in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, d))})
        total_dirs = len(year_dirs)

        for scanned_dirs, year_dir in enumerate(year_dirs, start=1):
            data_list = []
            print(f"Progress: {(scanned_dirs / total_dirs) * 100:.2f}% years processed.")

            for half in ['A', 'B']:
                half_year_dir = f"{year_dir}{half}"
                subdir_path = os.path.join(self.path, half_year_dir)

                if os.path.isdir(subdir_path):
                    for root, _, files in os.walk(subdir_path):
                        if any(file.endswith(self.quaternion_file_type) for file in files):
                            data_list += self._process_files(files, root)
            self._save_combined_tensor(data_list, f"{self.savefile_prefix}_year_{year_dir}.pt")
            del data_list

    def init_alldata_tensors(self):
        batch_size_limit = 2 * 1024 ** 3
        batch_data_list = []
        batch_current_size = 0
        save_path = f"{self.savefile_prefix}_all_data.pt"

        for root, dirs, files in os.walk(self.path):
            if any(file.endswith(self.quaternion_file_type) for file in files):
                data_list = self._process_files(files, root)
                batch_data_list.extend(data_list)
                batch_current_size += sum(arr.nbytes for arr in data_list)

                if batch_current_size >= batch_size_limit:
                    self._save_combined_tensor(batch_data_list, save_path)
                    batch_data_list.clear()
                    batch_current_size = 0

        if batch_data_list:
            self._save_combined_tensor(batch_data_list, save_path)

    def init_channel_tensors(self):
        channel_num = 6 if self.instruction == "HiCullGoodTimes.txt" else 8 if self.instruction == "LoGoodTimes.txt" else None
        if channel_num:
            for i in range(1, channel_num + 1):
                self._init_channel_tensor(i)
        else:
            print("Incorrect instruction file. Aborting...")

    def _init_channel_tensor(self, channel_index):
        batch_size_limit = 2 * 1024 ** 3
        batch_data_list = []
        batch_current_size = 0
        save_path = f"{self.savefile_prefix}_channel_{channel_index}.pt"
        channel_file_regex = f"{self.file_type}-{channel_index}"

        for root, dirs, files in os.walk(self.path):
            data_list = self._process_files([f for f in files if channel_file_regex in f], root)
            batch_data_list.extend(data_list)
            batch_current_size += sum(arr.nbytes for arr in data_list)

            if batch_current_size >= batch_size_limit:
                self._save_combined_tensor(batch_data_list, save_path)
                batch_data_list.clear()
                batch_current_size = 0

        if batch_data_list:
            self._save_combined_tensor(batch_data_list, save_path)

    def _process_files(self, files, root):
        data_list = []
        for file in files:
            if self.file_type in file:
                file_path = os.path.join(root, file)
                text = np.loadtxt(file_path, dtype='str')
                text = self.remove_or_convert_hex_flags(text)
                data_list.append(np.array(text, dtype=float))
        del text
        gc.collect()
        return data_list

    def _save_combined_tensor(self, data_list, save_path):
        if data_list:
            combined_data = np.vstack(data_list)
            torch.save(torch.tensor(combined_data), save_path)
            print(f"Saved tensor with shape {combined_data.shape} to {save_path}")
            del combined_data
        del data_list
        gc.collect()

    def remove_or_convert_hex_flags(self, data_list):
        if self.translate_hex:
            ch_column = data_list[:, 3]
            ty_column = data_list[:, 4]
            int_ch_column = np.vectorize(lambda x: int(x, 16))(ch_column)
            int_ty_column = np.vectorize(lambda x: int(x, 16))(ty_column)
            data_list[:, 3] = int_ch_column  # changing ch from hex to int
            data_list[:, 4] = int_ty_column  # changing ty from hex to int
            data_list[:,
            6] = '0'  # selnbits are not used so ve just ignore them (i dont know what they are and i dont care tbh)
        else:
            data_list[:, 3] = '0'
            data_list[:, 4] = '0'
            data_list[:, 6] = '0'  # still ignoring them
        gc.collect()
        return data_list


class ChannelAnalyzer:

    def __init__(self, config: str = "config.yml"):
        self.hi_tensor_directory = None
        self.lo_tensor_directory = None
        self.filename_prefix_hi = None
        self.filename_prefix_lo = None
        self.hi_instruction_file = None
        self.lo_instruction_file = None
        self.channel_labels = [f'hi{i}' for i in range(1, 7)] + [f'lo{i}' for i in range(1, 9)]
        self.pearson_matrix = np.zeros((14, 14))
        try:
            with open(config, 'r') as config_file:
                self.cfg = yaml.load(config_file, Loader=yaml.FullLoader)
        except FileNotFoundError as ex:
            print(f"Error: there is no such file as {config}. Exception: {ex}")

    def load_tensor(self, path):
        return torch.load(path)

    def set_analyzer_params(self):
        self.hi_tensor_directory = self.cfg["ChannelAnalyzer"]["hi_tensor_directory"]
        self.lo_tensor_directory = self.cfg["ChannelAnalyzer"]["lo_tensor_directory"]
        self.filename_prefix_hi = self.cfg["ChannelAnalyzer"]["filename_prefix_hi"]
        self.filename_prefix_lo = self.cfg["ChannelAnalyzer"]["filename_prefix_lo"]
        self.hi_instruction_file = self.cfg["ChannelAnalyzer"]["hi_instruction_file"]
        self.lo_instruction_file = self.cfg["ChannelAnalyzer"]["lo_instruction_file"]

    def translate_hex_to_int(self, hex_list):
        return [int(item, 16) for item in hex_list]

    def managing_data_based_on_instruction_files(self, tensor, tensor_name, instruction_file, is_hi_channel):
        if is_hi_channel:
            channel_num = int(re.search(r'channel_(.+?).pt', tensor_name).group(1))
            dtype = [('orbit', 'i4'), ('start_time', 'f8'), ('end_time', 'f8'), ('phase_start', 'i4'),
                     ('phase_end', 'i4'), ('dataset', 'U2'), ('channel_1', 'i4'), ('channel_2', 'i4'),
                     ('channel_3', 'i4'), ('channel_4', 'i4'), ('channel_5', 'i4'), ('channel_6', 'i4')]
        else:
            channel_num = int(
                re.search(r'channel_(.+?).pt', tensor_name).group(1)) - 6
            dtype = [('orbit', 'i4'), ('start_time', 'f8'), ('end_time', 'f8'), ('phase_start', 'i4'),
                     ('phase_end', 'i4'), ('dataset', 'U2'), ('channel_1', 'i4'), ('channel_2', 'i4'),
                     ('channel_3', 'i4'), ('channel_4', 'i4'), ('channel_5', 'i4'), ('channel_6', 'i4'),
                     ('channel_7', 'i4'), ('channel_8', 'i4')]

        instruction_data = np.genfromtxt(instruction_file, dtype=dtype, encoding=None)

        time_start_col = 1
        time_end_col = 2
        phase_start_col = 3
        phase_end_col = 4
        channel_bool_checker_col = 5 + channel_num

        good_data_intervals = []
        for row in instruction_data:
            if row[channel_bool_checker_col] == 1 and row[phase_start_col] == 0 and row[phase_end_col] == 59:
                start_time = row[time_start_col]
                end_time = row[time_end_col]
                good_data_intervals.append({'start_time': start_time, 'end_time': end_time})

        good_data_sums = []
        for interval in good_data_intervals:
            start_time = interval['start_time']
            end_time = interval['end_time']
            valid_data = tensor[(tensor[:, 0] >= start_time) & (tensor[:, 0] <= end_time)]
            sum_valid_data = valid_data[:, 5].sum().item()
            good_data_sums.append(sum_valid_data)
        gc.collect()
        return good_data_sums

    def weighted_pearsons_coefficient(self, X_vals, Y_vals, weights):
        if len(X_vals) == 0 or len(Y_vals) == 0:
            return np.nan
        X_vals = np.array(X_vals)
        Y_vals = np.array(Y_vals)
        weights = np.array(weights)

        mean_x = np.average(X_vals, weights=weights)
        mean_y = np.average(Y_vals, weights=weights)

        covariance = np.sum(weights * (X_vals - mean_x) * (Y_vals - mean_y))
        std_x = np.sqrt(np.sum(weights * (X_vals - mean_x) ** 2))
        std_y = np.sqrt(np.sum(weights * (Y_vals - mean_y) ** 2))

        if std_x == 0 or std_y == 0:
            return np.nan
        gc.collect()
        return covariance / (std_x * std_y)

    def calculate_pearsons_for_all_channels(self):
        for i in range(1, 15):
            for j in range(1, 15):
                if i <= 6:
                    tensor_i_path = os.path.join(self.hi_tensor_directory, f"{self.filename_prefix_hi}{i}.pt")
                    instruction_file_i = self.hi_instruction_file
                    is_hi_channel_i = True
                else:
                    tensor_i_path = os.path.join(self.lo_tensor_directory, f"{self.filename_prefix_lo}{i - 6}.pt")
                    instruction_file_i = self.lo_instruction_file
                    is_hi_channel_i = False

                if j <= 6:
                    tensor_j_path = os.path.join(self.hi_tensor_directory, f"{self.filename_prefix_hi}{j}.pt")
                    instruction_file_j = self.hi_instruction_file
                    is_hi_channel_j = True
                else:
                    tensor_j_path = os.path.join(self.lo_tensor_directory, f"{self.filename_prefix_lo}{j - 6}.pt")
                    instruction_file_j = self.lo_instruction_file
                    is_hi_channel_j = False

                try:
                    tensor_i = self.load_tensor(tensor_i_path)
                    tensor_j = self.load_tensor(tensor_j_path)

                    counts_i = self.managing_data_based_on_instruction_files(tensor_i, f"channel_{i}.pt",
                                                                             instruction_file_i, is_hi_channel_i)
                    counts_j = self.managing_data_based_on_instruction_files(tensor_j, f"channel_{j}.pt",
                                                                             instruction_file_j, is_hi_channel_j)

                    min_length = min(len(counts_i), len(counts_j))
                    counts_i = counts_i[:min_length]
                    counts_j = counts_j[:min_length]

                    weights = np.ones(min_length)

                    pearson_value = self.weighted_pearsons_coefficient(counts_i, counts_j, weights)
                    self.pearson_matrix[i - 1, j - 1] = pearson_value
                    print(f"Weighted Pearson coefficient for channels {i} and {j}: {pearson_value}")
                    del counts_i
                    del counts_j
                    del weights
                except FileNotFoundError as e:
                    print(f"File not found for channels {i} and {j}: {e}")
                except Exception as e:
                    print(f"Unexpected error for channels {i} and {j}: {e}")

        gc.collect()

    def save_matrix_to_file(self, filename):
        np.savetxt(filename, self.pearson_matrix, delimiter=',', header=','.join(self.channel_labels),
                   fmt='%.4f')


if __name__ == "__main__":
    tensor_creator = TensorCreator("config.yml")
    print("Loading configuration parameters...")
    tensor_creator.set_creation_params()
    print("Initializing tensor creation process...")
    tensor_creator.init_tensor_creation_process()
    tensor_analyzer = ChannelAnalyzer("config.yml")
    print("Loading analysis parameters...")
    tensor_analyzer.set_analyzer_params()
    tensor_analyzer.calculate_pearsons_for_all_channels()
