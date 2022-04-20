# Using pytorch-a2c-ppo-acktr-gail baselines in gym-miniworld

python main.py --env-name "MiniWorld-MazeS3Fast-v0" --algo a2c
python main.py --env-name "MiniWorld-MazeS3Fast-v0" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --num-steps 300 --num-mini-batch 4 --vis-interval 1 --log-interval 1

python main.py --env-name "MiniWorld-MazeS8-v0" --algo a2c
python main.py --env-name "MiniWorld-MazeS8-v0" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --num-steps 300 --num-mini-batch 4 --vis-interval 1 --log-interval 1