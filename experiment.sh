# start tmux sessions one for each experiment
tmux new -s standard_maml_exp -d
tmux new -s pq_maml_exp -d

# initialise virtual environments
tmux send-keys -t "standard_maml_exp" "source /Users/sebastianlee/Dropbox/Documents/Work/Hack/Environments/meta/bin/activate" C-m
tmux send-keys -t "pq_maml_exp" "source /Users/sebastianlee/Dropbox/Documents/Work/Hack/Environments/meta/bin/activate" C-m

# send keys to run experiments
tmux send-keys -t "standard_maml_exp" "python main.py maml_config.yaml" C-m
tmux send-keys -t "pq_maml_exp" "python main.py pq_maml_config.yaml" C-m

# start tmux session for tensorboard, launch tensorboard
tmux new -s tensorboard -d
tmux send-keys -t "tensorboard" "tensorboard --logdir /Users/sebastianlee/Dropbox/Documents/Work/Hack/MAML_project/maml_seb/runs/None/" C-m



