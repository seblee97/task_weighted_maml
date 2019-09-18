# kill sessions already running
tmux kill-session -t pq_epsilon_maml_exp
tmux kill-session -t standard_maml_exp
tmux kill-session -t pq_sample_maml_exp
tmux kill-session -t tensorboard

# start tmux sessions one for each experiment
tmux new -s standard_maml_exp -d
tmux new -s pq_epsilon_maml_exp -d
tmux new -s pq_sample_maml_exp -d

# initialise virtual environments
tmux send-keys -t "standard_maml_exp" "source ~/envs/meta/bin/activate" C-m
tmux send-keys -t "pq_epsilon_maml_exp" "source ~/envs/meta/bin/activate" C-m
tmux send-keys -t "pq_sample_maml_exp" "source ~/envs/meta/bin/activate" C-m

# send keys to run experiments
tmux send-keys -t "standard_maml_exp" "python main.py -config configs/maml_config.yaml" C-m
tmux send-keys -t "pq_epsilon_maml_exp" "python main.py -config configs/pq_maml_config.yaml" C-m
tmux send-keys -t "pq_sample_maml_exp" "python main.py -config configs/pq_sample_maml_config.yaml" C-m

# start tmux session for tensorboard, launch tensorboard
tmux new -s tensorboard -d
tmux send-keys -t "tensorboard" "source ~/envs/meta/bin/activate" C-m
tmux send-keys -t "tensorboard" "tensorboard --logdir results/" C-m
