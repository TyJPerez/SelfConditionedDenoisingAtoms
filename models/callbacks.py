

import torch 
import numpy as np
import os
import time
from tqdm import tqdm
import wandb

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback


import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from scipy.stats import spearmanr
from scipy.stats import pearsonr


def compute_corr(predictions, targets):
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()

    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()

    spearman_corr, p_spearman = spearmanr(predictions, targets)
    rmse = np.sqrt(np.mean((predictions-targets)**2))
    pearson_corr, p_pearson = pearsonr(predictions.flatten(), targets.flatten())
    
    return pearson_corr, spearman_corr, rmse
    # return {
    #     # "rmse": rmse,
    #     "pearson_corr": pearson_corr,
    #     "spearman_corr": spearman_corr,
    #     # "p_pearson": p_pearson,
    #     # "p_spearman": p_spearman  
    # }

class ParityPlot(Callback):
    """Callback to call parity plot at the end of training
    
    args:
        output_dir: str, directory to save plots
        end_stages: list of str, stages to plot at the end of training (default: ['train', 'test'])
        stages: list of str, stages to plot during training (default: ['test'])
        plot_interval: int, interval in epochs to plot parity plots during training. if <1, no plots during training (default: -1)
    
    """
    def __init__(self, 
                 output_dir: str = None, 
                 end_stages: list = ['train', 'test'], # what stages to plot at the end of training
                 stages: list = ['test'],  # what stages to plot during training
                 plot_interval: int = -1, # in epochs, the interval to plot parity plots during training. if <1, no plots during training
                 ):
        super().__init__()
        self.output_dir = output_dir
        self.stages = stages
        self.end_stages = end_stages
        self.plot_interval = plot_interval # in epochs

    # def on_train_end(self, trainer, pl_module):
    def on_fit_end(self, trainer, pl_module):
        for stage in self.end_stages:
            self.parity_plot(trainer, pl_module, stage=stage)
    
    def on_train_epoch_end(self, trainer, pl_module):
        if self.plot_interval < 1:
            return
        #plot parity plot at the end of validation every N epochs
        current_epoch = pl_module.current_epoch
        if (current_epoch + 1) % self.plot_interval == 0:
            for stage in self.stages:
                self.parity_plot(trainer, pl_module, stage=stage)
    
    # def on_fit_start(self, trainer, pl_module):
        # return
        # for stage in self.stages:
        # for stage in self.end_stages: #Temp debug
        #     self.parity_plot(trainer, pl_module, stage=stage)
    
    def parity_plot(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str = 'test'):

        if trainer.world_size > 1:
            #return if not the main process
            if trainer.global_rank != 0:
                return

        #check if model is in pretrain mode
        if pl_module.hparams.get('pretrain_mode', False):
            print("Model in pretrain mode, skipping parity plot.")
            return
        
        trainer.strategy.model_to_device()
        device = pl_module.device
        print(f"Generating parity plot for {stage} set on device {device}")

        #FIXE THE DROPPATH PROBLEM
        # dp_stat = pl_module.model.rep_model.attention_layers[0].joint_droppath.drop_prob
        # print(f"----- Current droppath rate: {dp_stat} -----")
        
        # set moodel to eval mode
        pl_module.eval()

        dmod = trainer.datamodule
        # Use provided datamodule
        if stage == 'train':
            dataloader = dmod.train_dataloader()
        elif stage == 'val':
            dataloader = dmod.val_dataloader()
            if isinstance(dataloader, list):
                dataloader = dataloader[0]  # Handle multiple val dataloaders
        elif stage == 'test':
            dataloader = dmod.test_dataloader()
        else:
            raise ValueError(f"Invalid stage: {stage}. Must be 'train', 'val', or 'test'")
        
        predictions = []
        targets = []
        nodes_per_graph_list = []
        batch_indices_list = []  # Track batch indices for per-node predictions

        #the model will use the internal test step for all parity plots
        internal_stage_flag = 'test' 
        
        # Collect all predictions without gradients
        loss_fn = torch.nn.MSELoss() # Dummy loss function for step method
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Generating parity plot for {stage} set"):
                batch = trainer.strategy.batch_to_device(batch, device)
                output_dict, batch = pl_module.step(batch, loss_fn, internal_stage_flag, return_outputs=True)
                #keys: 'y', 'noise_pred', 'dy' (if derivative is on)

                #compute number of nodes in each graph
                # if hasattr(batch, 'batch'):
                batch_indices = batch.batch.cpu().numpy()
                num_graphs = batch_indices.max() + 1
                nodes_per_graph = np.bincount(batch_indices, minlength=num_graphs)
                nodes_per_graph_list.extend(nodes_per_graph.tolist())

                #check if derivative is present
                dy_pred = output_dict.get('dy', None)
                if pl_module.hparams.derivative and dy_pred is not None:
                    # raise NotImplementedError("Parity plot for derivative prediction not implemented yet.")
                    print("Warning: Derivative predictions present, but parity plot for derivatives not implemented. Skipping derivative parity.")
                    
                # Extract predictions and targets
                pred = output_dict['y']
                
                # Handle case where targets might not be available
                if hasattr(batch, 'y') and batch.y is not None:
                    target = batch.y
                    
                    # Ensure proper dimensionality
                    if pred.ndim > 1 and pred.shape[1] == 1:
                        pred = pred.squeeze(1)
                    if target.ndim > 1 and target.shape[1] == 1:
                        target = target.squeeze(1)
                    
                    predictions.append(pred.cpu())
                    targets.append(target.cpu())
                
                elif hasattr(batch, 'noise') and batch.noise is not None:
                    # If batch has noise, use it as target (e.g., for self-conditioned models)
                    target = batch.noise
                    pred = output_dict.get('noise_pred', None)
                    
                    predictions.append(pred.cpu())
                    targets.append(target.cpu())
                    # Store batch indices for per-node coloring
                    batch_indices_list.append(batch.batch.cpu())

                # if len(nodes_per_graph_list) >= max_samples:
                #     break  # Limit to max_samples for plotting
        
        if len(predictions) == 0:
            print(f"No predictions collected for stage '{stage}'. Skipping parity plot.")
            return
        
        # Concatenate all predictions and targets
        predictions = torch.cat(predictions, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()
        
        # Prepare nodes_per_graph for coloring
        nodes_per_graph_flat = np.array(nodes_per_graph_list)
        
        # If we have per-node predictions, expand nodes_per_graph to match
        if len(batch_indices_list) > 0:
            # Per-node predictions (e.g., 3D noise vectors)
            # Expand node counts so each node gets its graph's count as color value
            nodes_per_node = []
            for node_count in nodes_per_graph_list:
                nodes_per_node.extend([node_count] * node_count)
            nodes_per_graph_array = np.array(nodes_per_node)
            # If predictions are multi-dimensional (e.g., shape (N, 3)), scatter flattens
            # them to N*3 points, so we need to repeat the color array accordingly
            if predictions.ndim > 1:
                nodes_per_graph_array = np.repeat(nodes_per_graph_array, predictions.shape[1])
        else:
            nodes_per_graph_array = nodes_per_graph_flat
        
        # Compute stats from per-graph counts (not expanded)
        max_nodes_per_graph = nodes_per_graph_flat.max()
        min_nodes_per_graph = nodes_per_graph_flat.min()
        mean_nodes_per_graph = nodes_per_graph_flat.mean()
        std_nodes_per_graph = nodes_per_graph_flat.std()
        vmin = nodes_per_graph_flat.min()
        vmax = mean_nodes_per_graph + 2 * std_nodes_per_graph
        print(f"Nodes per graph - min: {min_nodes_per_graph}, max: {max_nodes_per_graph}, mean: {mean_nodes_per_graph:.2f}, std: {std_nodes_per_graph:.2f}" )

        # Flatten multi-dimensional predictions (e.g., 3D noise) for scalar metrics
        predictions_flat = predictions.flatten()
        targets_flat = targets.flatten()

        # Calculate metrics
        r2 = r2_score(targets_flat, predictions_flat)
        mae = mean_absolute_error(targets_flat, predictions_flat)
        pearson_corr, spearman_corr, rmse = compute_corr(predictions_flat, targets_flat)

        
        # Create parity plot
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Scatter plot - colored by nodes per graph
        sc = ax.scatter(targets_flat, predictions_flat, c=nodes_per_graph_array, cmap='viridis', alpha=0.5, s=15, edgecolors='none', vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(sc)
        cbar.set_label('Number of Nodes per Graph')
        
        # Perfect prediction line (y=x)
        min_val = min(targets_flat.min(), predictions_flat.min())
        max_val = max(targets_flat.max(), predictions_flat.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Add metrics to plot
        # textstr = f'R² = {r2:.3f}\nMAE = {mae:.3f}\nRMSE = {rmse:.3f}'
        textstr = (f'R² = {r2:.3f}\nMAE = {mae:.3f}\nRMSE = {rmse:.3f}\n'
                   f'Pearson r = {pearson_corr:.3f}\nSpearman r = {spearman_corr:.3f}')
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        
        # Formatting

        #include current step in title
        title_text = f'Parity Plot - {stage.capitalize()} Set (Step {pl_module.global_step})'
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(title_text)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Make axes equal
        ax.set_aspect('equal', adjustable='box')
        
        # Set equal limits
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        
        plt.tight_layout()

        #save figure locally
        output_dir = self.output_dir #trainer.hparams.log_dir #, self.hparams.job_id)
        filename = f'{pl_module.hparams.job_id}_parity-plot_{stage}_ep{pl_module.current_epoch}.png'
        fig.savefig(os.path.join(output_dir, filename), dpi=300)

        # # Log to wandb
        if wandb.run is not None:
            wandb.log({
                f"parity_plot_{stage}": wandb.Image(fig),
                f"r2_{stage}": r2,
                f"mae_{stage}": mae,
                f"rmse_{stage}": rmse,
                f"pearson_corr_{stage}": pearson_corr,
                f"spearman_corr_{stage}": spearman_corr,
                "epoch": pl_module.current_epoch
            })
            print(f"Parity plot for {stage} logged to wandb (R²={r2:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}, Pearson r={pearson_corr:.3f}, Spearman r={spearman_corr:.3f})")
        else:
            print(f"wandb not initialized. Parity plot created but not logged.")
            print(f"Metrics - R²: {r2:.3f}, MAE: {mae:.3f}, RMSE: {rmse:.3f}")
        
        plt.close(fig)  # Close to free memory


class LimitRun(Callback):
    """Stop *this run only* after `minutes` from start (no accumulation across restarts).
    Saves a checkpoint before stopping so the next job can resume.
    """
    def __init__(self, 
                 minutes : int = None, 
                 max_epochs_per_run: int = None,
                 out_dir: str = None, 
                 buffer_sec: int = 30,
                  ):
        super().__init__()
        self.minutes = minutes
        self.out_dir = out_dir
        self.buffer_sec = buffer_sec
        self.max_epochs_per_run = max_epochs_per_run
        self._t0 = None
        self._start_epoch = None

    def on_fit_start(self, trainer, pl_module):
        # This is called before checkpoint loading
        self._t0 = time.time()
        os.makedirs(os.path.join(self.out_dir), exist_ok=True)

    def on_train_start(self, trainer, pl_module):
        # This is called AFTER checkpoint loading
        self._start_epoch = trainer.current_epoch

        current_steps = trainer.global_step
        if current_steps < 1:
            self._start_epoch = -1 

        # Print what limits are active
        limits = []
        if self.minutes is not None:
            limits.append(f"{self.minutes} minutes")
        if self.max_epochs_per_run is not None:
            limits.append(f"{self.max_epochs_per_run} epochs")

        if len(limits) > 0 and trainer.global_rank == 0:
            print(f"Will stop this run after: {' OR '.join(limits)} from start epoch {self._start_epoch}")
    
    def get_filename(self, trainer):
        step = trainer.global_step
        epoch = trainer.current_epoch
        
        filename = f"step={step}-epoch={epoch}.stop_ckpt"
        return filename

    def _save_checkpoint_and_stop(self, trainer, reason="time limit"):
        """Helper method to save checkpoint and stop training"""
        
        ckpt_dir = os.path.join(self.out_dir)
        tmp_path = os.path.join(ckpt_dir, self.get_filename(trainer))
        # trainer.save_checkpoint(tmp_path)

        # if trainer.global_rank == 0:  # Only rank 0 saves
            # Optionally update last.ckpt
            # last_path = os.path.join(ckpt_dir, "last.ckpt")
            # shutil.copy2(tmp_path, last_path)

        model_checkpoint_call_found = False
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                # callback._save_last_checkpoint(tmp_path)
                
                callback._save_topk_checkpoint(trainer, trainer.logged_metrics)
                # Also save last checkpoint if enabled
                if callback.save_last:
                    callback._save_last_checkpoint(trainer, trainer.logged_metrics)

                model_checkpoint_call_found = True
        if not model_checkpoint_call_found:
            print('WARNING: no model checkpoint callback found!! check point was not saved. no manual save process set up')
        
        print(f"--- STOPING TRAINING: due to {reason}. Saved checkpoint: {tmp_path}")
        
        trainer.should_stop = True  # graceful stop at the next safe point
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Check time limit during training"""
        if self._t0 is None or self.minutes is None:
            return
            
        elapsed = time.time() - self._t0
        limit = max(0, 60 * self.minutes - self.buffer_sec)
        if elapsed >= limit:
            self._save_checkpoint_and_stop(trainer, "time limit")
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Check epoch limit at the end of each epoch"""
        if self._start_epoch is None or self.max_epochs_per_run is None:
            return
            
        epochs_completed = trainer.current_epoch - self._start_epoch #+ 1
        # print(f"Completed epochs: {epochs_completed}, current epoch {trainer.current_epoch}, start epoch, {self._start_epoch}")
        if epochs_completed >= self.max_epochs_per_run:
            self._save_checkpoint_and_stop(trainer, f"epoch limit ({epochs_completed} epochs completed)")
    
    # def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    #     # Check elapsed wall time from the start of *this* run
    #     if self._t0 is None:
    #         return
    #     elapsed = time.time() - self._t0
    #     limit = max(0, 60 * self.minutes - self.buffer_sec)
    #     if elapsed >= limit:
    #         # Save an emergency checkpoint and request a clean stop
    #         ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    #         # ckpt_dir = os.path.join(self.out_dir, "checkpoints")
    #         ckpt_dir = os.path.join(self.out_dir)
    #         # tmp_path = os.path.join(ckpt_dir, f"last_runstop_{ts}.temp_ckpt")
    #         tmp_path = os.path.join(ckpt_dir, self.get_filename(trainer))
    #         trainer.save_checkpoint(tmp_path)
    #         # also refresh/overwrite a stable "last.ckpt" pointer
    #         last_path = os.path.join(ckpt_dir, "last.ckpt")
    #         # try:
    #         #     if os.path.islink(last_path) or os.path.exists(last_path):
    #         #         os.remove(last_path)
    #         #     os.symlink(os.path.basename(tmp_path), last_path)
    #         # except Exception:
    #         #     # fall back to copy if symlink not allowed
    #         #     import shutil
    #         #     shutil.copy2(tmp_path, last_path)
    #         # shutil.copy2(tmp_path, last_path)

    #         trainer.should_stop = True  # graceful stop at the next safe point



def _save_checkpoint(out_path, trainer, reason="time limit"):
        """Helper method to save checkpoint and stop training"""
        
        # ckpt_dir = os.path.join(self.out_dir)
        # tmp_path = os.path.join(ckpt_dir, self.get_filename(trainer))

        model_checkpoint_call_found = False
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                # callback._save_last_checkpoint(tmp_path)
                
                callback._save_topk_checkpoint(trainer, trainer.logged_metrics)
                # Also save last checkpoint if enabled
                if callback.save_last:
                    callback._save_last_checkpoint(trainer, trainer.logged_metrics)

                model_checkpoint_call_found = True
        
        if not model_checkpoint_call_found:
            print('WARNING: no model checkpoint callback found!! check point was not saved. no manual save process set up')
        
        print(f"--- STOPING TRAINING: due to {reason}. Saved checkpoint: {tmp_path}")
        
        trainer.should_stop = True  # graceful stop at the next safe point