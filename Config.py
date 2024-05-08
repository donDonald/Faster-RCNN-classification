import os
import sys
import torch


def eprint(*args, **kwargs):
        print(*args, file=sys.stderr, **kwargs)




class Config:

        def __init__(self, args):
                # verbosity level
                self._verbosity = args.verbosity

                # Training images and XML files directory
                self._train_dir = args.train_dir
                if self.verbosity > 0:
                        print(f'training data directory:{self.train_dir}')

                if not os.path.isdir(self.train_dir):
                        eprint(f'train_dir "{self.train_dir}" does not exist')
                        eprint('exiting')
                        sys.exit(1)

                # Validation images and XML files directory
                self._valid_dir = args.valid_dir
                if self.verbosity > 0:
                        print(f'validation data directory:{self.valid_dir}')

                if not os.path.isdir(self.valid_dir):
                        eprint(f'valid_dir "{self.valid_dir}" does not exist')
                        eprint('exiting')
                        sys.exit(1)

                # classes: 0 index is reserved for background
                train_list = next(os.walk(self.train_dir))[1]
                valid_list = next(os.walk(self.valid_dir))[1]
                if train_list != valid_list:
                        eprint(f'training and validation classes do not match\n    training:{train_list}\n    validation:{valid_list}')
                        eprint('exiting')
                        sys.exit(1)

                if self.verbosity > 0:
                        print(f'classes to be trained:{train_list}')

                self._classes = ['background'] + train_list

                # Location to save model and plots
                self._out_dir = args.out_dir
                if not os.path.isdir(self.out_dir):
                        eprint(f'out_dir "{self.out_dir}" does not  exist')
                        eprint('exiting')
                        sys.exit(1)
                if self.verbosity > 0:
                        print(f'output directory:{self.out_dir}')

                # Rest of the options
                self._save_plots_epoch = args.save_plots_epoch
                if self.verbosity > 0:
                        print(f'plots will be saved every:{self.save_plots_epoch} epoch')

                self._save_model_epoch = args.save_model_epoch
                if self.verbosity > 0:
                        print(f'model will be saved every:{self.save_model_epoch} epoch')

                self._batch_size = args.batch_size
                if self.verbosity > 0:
                        print(f'batch size:{self.batch_size}')

                self._resize_to = args.resize_to
                if self.verbosity > 0:
                        print(f'resize to:{self.resize_to}')

                self._num_epochs = args.num_epochs
                if self.verbosity > 0:
                        print(f'train for epochs:{self.num_epochs}')

                self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                if self.verbosity > 0:
                        print(f'using device:{self.device}')

                #Additional Info when using cuda
                if self.verbosity > 0 and self.device.type == 'cuda':
                        print(torch.cuda.get_device_name(0))
                        print('memory Usage:')
                        print('allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
                        print('cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')


        def __str__(self):
                return f"""
                    verbosity:{self.verbosity}
                    train_dir:{self.train_dir}
                    valid_dir:{self.valid_dir}
                    out_dir:{self.out_dir}
                    classes:{self.classes}
                    num_classes:{self.num_classes}
                    num_epochs:{self.num_epochs}
                    save_plots_epoch:{self.save_plots_epoch}
                    save_model_epoch:{self.save_model_epoch}
                    batch_size:{self.batch_size}
                    resize_to:{self.resize_to}
                """


        #@verbosity.getter
        @property
        def verbosity(self):
                return self._verbosity


        #@train_dir.getter
        @property
        def train_dir(self):
                return self._train_dir


        #@valid_dir.getter
        @property
        def valid_dir(self):
                return self._valid_dir


        #@out_dir.getter
        @property
        def out_dir(self):
                return self._out_dir


        #@save_plots_epoch.getter
        @property
        def save_plots_epoch(self):
                return self._save_plots_epoch


        #@save_model_epoch.getter
        @property
        def save_model_epoch(self):
                return self._save_model_epoch


        #@batch_size.getter
        @property
        def batch_size(self):
                return self._batch_size


        #@resize_to.getter
        @property
        def resize_to(self):
                return self._resize_to


        #@num_epochs.getter
        @property
        def num_epochs(self):
                return self._num_epochs


        #@classes.getter
        @property
        def classes(self):
                return self._classes


        #@num_classes.getter
        @property
        def num_classes(self):
                return len(self._classes)


        #@device.getter
        @property
        def device(self):
                return self._device
