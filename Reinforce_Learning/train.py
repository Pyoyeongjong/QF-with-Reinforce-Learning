from a2c import A2Cagent
from chartenv import ChartTradingEnv
import data
import parameter
import time as TIME
import numpy as np
import tensorflow as tf
import os
import pandas as pd

DEBUG = False
TIMEDEBUG = False

LONG_TYPE = 0
SHORT_TYPE = 1
ALL_TYPE = 2
BASE_DIR = parameter.BASE_DIR
tickers = parameter.tickers
GAMMA = parameter.GAMMA
BATCH_SIZE = parameter.BATCH_SIZE

class Train:

    def __init__(self, time_steps, agent_type, tickers, budget):
        self.env = ChartTradingEnv(time_steps, tickers, budget)
        # 이 에이전트는 어떤 행동을 할 것인가?
        self.agent_type = agent_type
        if agent_type == LONG_TYPE:
            self.TradeAgent = A2Cagent(time_steps, self.env.observation_space.shape[0], 2, GAMMA, BATCH_SIZE) # 0 이 매수, 1 이 관망
            self.LongAgent = A2Cagent(time_steps, self.env.observation_space.shape[0], 2, GAMMA, BATCH_SIZE)
        elif agent_type == SHORT_TYPE:
            self.TradeAgent = A2Cagent(time_steps, self.env.observation_space.shape[0], 2, GAMMA, BATCH_SIZE)
            self.ShortAgent = A2Cagent(time_steps, self.env.observation_space.shape[0], 2, GAMMA, BATCH_SIZE)
        else:
            self.TradeAgent = A2Cagent(time_steps, self.env.observation_space.shape[0], 3, GAMMA, BATCH_SIZE)
            self.LongAgent = A2Cagent(time_steps, self.env.observation_space.shape[0], 2, GAMMA, BATCH_SIZE)
            self.ShortAgent = A2Cagent(time_steps, self.env.observation_space.shape[0], 2, GAMMA, BATCH_SIZE)

        self.BATCH_SIZE = parameter.BATCH_SIZE_T
        self.BATCH_SIZE_LS = parameter.BATCH_SIZE_LS
        self.decay = parameter.INIT_DECAY
        # long ,short agent 용 batch들
        self.lstates, self.lactions, self.ltd_targets, self.ladvantages = [], [], [], []
        self.sstates, self.sactions, self.std_targets, self.sadvantages = [], [], [], []
        self.save_epi_reward = []
        # Train된 횟수
        self.tradetrain = 0
        self.longtrain = 0
        self.shorttrain = 0
        # print용
        self.printnum = 0
        print("Trainer OK")

    
    def load_weights(self, path):
        self.TradeAgent.actor.model.load_weights(f"{path}/Trade_actor.weights.h5")
        self.TradeAgent.critic.model.load_weights(f"{path}/Trade_critic.weights.h5")
        if self.agent_type != SHORT_TYPE:
            self.LongAgent.actor.model.load_weights(f"{path}/Long_actor.weights.h5")
            self.LongAgent.critic.model.load_weights(f"{path}/Long_critic.weights.h5")
        if self.agent_type != LONG_TYPE:
            self.ShortAgent.actor.model.load_weights(f"{path}/Short_actor.weights.h5")
            self.ShortAgent.critic.model.load_weights(f"{path}/Short_critic.weights.h5")
        return
    

    def train(self, max_episode_num, curr_tick):

        print("Train start")
        self.env.curr_ticker = curr_tick % len(tickers)
        # ep 한 번 = 한 ticker를 한바퀴 돌았을 때
        for ep in range(curr_tick, (int(max_episode_num))):
            # initialize batch
            states, actions, td_targets, advantages = [], [], [], []
            # 에피소드 리셋
            time, episode_reward, ticker_done = 0, 0, False
            # 소요시간
            start = TIME.time()
            # Env 첫 state 가져오기
            _, state = self.env.reset(test=False)
            lowest_budget = self.env.budget
            highest_budget = self.env.budget

            while not ticker_done:
                ta_act = self.TradeAgent.get_action(state)
                act = np.random.choice(len(ta_act), p=ta_act)

                # 판단할 수 없을 때는 reward가 반드시 None이다.
                next_state, reward, ticker_done, info = self.action_step(act)
                # 종료 판단
                # ticker_done이 True면 차트 특성상 가치평가를 할 수 없다.
                if ticker_done == True: # Ticker가 종료되었다.
                    break
                # 종료되지 않고 판단할 수 없다 -> 다음 행을 검토한다.
                # 이런 상황이 나와선 안된다. 결측지 문제임.
                if reward is None: 
                    _, state = self.env.get_next_row_obs()
                    continue
                # 최저 budget, 최고 budget 갱신
                if lowest_budget > self.env.budget:
                    lowest_budget = self.env.budget
                if highest_budget < self.env.budget:
                    highest_budget = self.env.budget
                # debug용
                if act < 2:
                    self.printnum += 1
                    if self.printnum % 100 == 0:
                        print(f"Time: {time}, Action: {act}, Reward: {reward*100:.2f}, Open: {info[0]:.5f}, Close: {info[1]:.5f}, Current: {info[2]:05}, Budget: {self.env.budget:.2f}")
                ### TradeAgent학습 시작
                # 모델에서 필요한 값 뽑아내기
                v_value = self.TradeAgent.critic.model(tf.convert_to_tensor(np.array([state]), dtype=tf.float32))
                next_v_value = self.TradeAgent.critic.model(tf.convert_to_tensor(np.array([next_state]), dtype=tf.float32))
                train_reward = reward
                advantage, y_i = self.TradeAgent.td_target(train_reward, v_value, next_v_value, ticker_done)
                # batch에 쌓기
                states.append(state)
                actions.append(ta_act)
                td_targets.append(y_i)
                advantages.append(advantage)
                # Model 학습
                if len(states) == self.BATCH_SIZE:
                    print("***********Trade_learning*************")
                    start = TIME.time()
                    # critic 학습
                    self.TradeAgent.critic_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                                           tf.convert_to_tensor(td_targets,dtype=tf.float32))
                    # actor 학습
                    self.TradeAgent.actor_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                                               tf.convert_to_tensor(actions, dtype=tf.float32),
                                           tf.convert_to_tensor(advantages,dtype=tf.float32))
                    self.tradetrain += 1
                    # batch 지우기
                    states, actions, td_targets, advantages = [], [], [], []
                    if TIMEDEBUG:
                        print("TradeAgent_learn exec time=",TIME.time() - start)

                episode_reward += reward
                state = next_state
                time += 1
            # while 밖
            # display each episode
            print('Episode: ', ep+1, 'Time: ', time, 'Reward: ', episode_reward)
            self.save_epi_reward.append(episode_reward)
            ## 각 weight 저장하기
            # 저장 코드
            save_weight_path = os.path.join(BASE_DIR, "save_weights")
            self.TradeAgent.actor.model.save_weights(f"{save_weight_path}/Trade_actor{ep}.weights.h5")
            self.TradeAgent.critic.model.save_weights(f"{save_weight_path}/Trade_critic{ep}.weights.h5")
            if self.agent_type != SHORT_TYPE:
                self.LongAgent.actor.model.save_weights(f"{save_weight_path}/Long_actor{ep}.weights.h5")
                self.LongAgent.critic.model.save_weights(f"{save_weight_path}/Long_critic{ep}.weights.h5")
            if self.agent_type != LONG_TYPE:
                self.ShortAgent.actor.model.save_weights(f"{save_weight_path}/Short_actor{ep}.weights.h5")
                self.ShortAgent.critic.model.save_weights(f"{save_weight_path}/Short_critic{ep}.weights.h5")
            print("[Save Weight]: ep",ep," save Completed")
            # Weights Text 저장
            save_weight_text = os.path.join(save_weight_path, "train_reward_txt")
            with open(save_weight_text, 'a') as f:
                f.write(f"ep{ep} Ticker: {tickers[self.env.curr_ticker]}, Reward: {episode_reward}, Budget: {self.env.budget}, LowB: {lowest_budget}, HighB: {highest_budget}\n")
        
        # for 밖
        print("Train Finished")

    # obs 가 none이 불리면, agent가 reset을 진행.
    def action_step(self, action):
        ticker_done = False
        info = ["None", "None", self.env.curr]
        #### Long/Short Action
        if action == 0 or action == 1:
            self.decay = parameter.INIT_DECAY
            # 다음 obs 가져오기
            ohlcv, state = self.env.get_next_row_obs()
            # obs None인지 확인
            if state is None:
                # ticker가 끝났는가?
                if self.env.ticker_is_done():
                    ticker_done = True # ticker끝
                else:
                    ticker_done = False
                info = ["Obs is none"]
                return None, None, ticker_done, info # None, None, True, _
            # 거래 변수들
            reward = 0      # 손익률
            position = ohlcv['open'] # obs open value
            close_position = None
            while(1): # Action이 종료될 때 까지
                # Raw Action 뽑아내기
                if action == 0:
                    raw_act = self.LongAgent.get_action(state)
                else:
                    raw_act = self.ShortAgent.get_action(state)
                act = np.random.choice(len(raw_act), p=raw_act)
                # obs가 None이거나, 거래거 끝났을 때 done = True
                next_state, reward, step_done, finish_info = self.env.long_or_short_step(act, position, action)
                # next_obs None = 판단불가능 (ticker가 끝났을 때 밖에 없어야 한다)
                if next_state is None:
                    # ticker가 끝났는가?
                    if self.env.ticker_is_done():
                        ticker_done = True # ticker끝
                    else:
                        ticker_done = False
                    info = ["obs is none"]
                    return None, None, ticker_done, info # None, None, True, _
                # reward(percent) -> reward로 바꿔주기
                reward = parameter.cal_reward(reward)
                ### Long Agent학습
                if action == 0:
                    # value 뽑아내기
                    v_value = self.LongAgent.critic.model(tf.convert_to_tensor(np.array([state]), dtype=tf.float32))
                    next_v_value = self.LongAgent.critic.model(tf.convert_to_tensor(np.array([next_state]), dtype=tf.float32))
                    train_reward = reward
                    advantage, y_i = self.LongAgent.td_target(train_reward, v_value, next_v_value, step_done)
                    # batch에 쌓기
                    self.lstates.append(state)
                    self.lactions.append(raw_act)
                    self.ltd_targets.append(y_i)
                    self.ladvantages.append(advantage)
                    # 학습하기
                    if len(self.lstates) == self.BATCH_SIZE_LS:
                        # if True: 배우는 비율 맞추기 self.tradetrain * 10 > self.longtrain
                        print("***********Long_learning*************")
                        start = TIME.time()
                        # critic 학습
                        self.LongAgent.critic_learn(tf.convert_to_tensor(self.lstates, dtype=tf.float32),
                                            tf.convert_to_tensor(self.ltd_targets, dtype=tf.float32))
                        # actor 학습
                        self.LongAgent.actor_learn(tf.convert_to_tensor(self.lstates, dtype=tf.float32),
                                                tf.convert_to_tensor(self.lactions, dtype=tf.float32),
                                            tf.convert_to_tensor(self.ladvantages,dtype=tf.float32))
                        if TIMEDEBUG:
                            print("LongAgent_learn exec time=",TIME.time() - start)
                        self.longtrain += 1
                        # batch 지우기
                        self.lstates, self.lactions, self.ltd_targets, self.ladvantages = [], [], [], []
                    # state 갱신
                    state = next_state
                    # 거래가 종료되었을 때
                    if step_done:
                        close_position = finish_info[0]
                        # Trade Agent를 위해 따로 reward를 계산해줘야 한다.
                        percent = self.env.cal_percent(position, close_position) - parameter.LONG_TRANS_FEE * 2
                        self.env.budget *= (1+percent)
                        reward = parameter.cal_reward(percent)
                        break # long 거래가 끝났으므로 빠져나간다.
                ### Short Agent학습
                else:
                    # value 뽑아내기
                    v_value = self.ShortAgent.critic.model(tf.convert_to_tensor(np.array([state]), dtype=tf.float32))
                    next_v_value = self.ShortAgent.critic.model(tf.convert_to_tensor(np.array([next_state]), dtype=tf.float32))
                    train_reward = reward
                    advantage, y_i = self.ShortAgent.td_target(train_reward, v_value, next_v_value, step_done)
                    # batch에 쌓기
                    self.sstates.append(state)
                    self.sactions.append(raw_act)
                    self.std_targets.append(y_i)
                    self.sadvantages.append(advantage)
                    # 학습하기
                    if len(self.sstates) == self.BATCH_SIZE_LS:
                        # if True: 배우는 비율 맞추기 self.tradetrain * 10 > self.train
                        print("***********Short_learning*************")
                        start = TIME.time()
                        # critic 학습
                        self.ShortAgent.critic_learn(tf.convert_to_tensor(self.sstates, dtype=tf.float32),
                                            tf.convert_to_tensor(self.std_targets, dtype=tf.float32))
                        # actor 학습
                        self.ShortAgent.actor_learn(tf.convert_to_tensor(self.sstates, dtype=tf.float32),
                                                tf.convert_to_tensor(self.sactions, dtype=tf.float32),
                                            tf.convert_to_tensor(self.sadvantages,dtype=tf.float32))
                        if TIMEDEBUG:
                            print("ShortAgent_learn exec time=",TIME.time() - start)
                        self.shorttrain += 1
                        # batch 지우기
                        self.sstates, self.sactions, self.std_targets, self.sadvantages = [], [], [], []
                    # state 갱신
                    state = next_state
                    # 거래가 종료되었을 때
                    if step_done:
                        close_position = finish_info[0]
                        # Trade Agent를 위해 따로 reward를 계산해줘야 한다.
                        percent = -self.env.cal_percent(position, close_position) - parameter.LONG_TRANS_FEE * 2
                        self.env.budget *= (1+percent)
                        reward = parameter.cal_reward(percent)
                        break # long 거래가 끝났으므로 빠져나간다.
            # Info 설정 후 return
            obs = state
            info = [position, close_position, self.env.curr]
            return obs, reward, ticker_done, info
        # 관망(2)이면 next_obs 그냥 다음 행 주기.
        else:
            _, obs = self.env.get_next_row_obs()
            if obs is None: # 판단할 수 없음. 버려야 함.
                reward = None
                tiekr_done = self.env.ticker_is_done()
            else:
                self.decay = self.decay * parameter.STAY_DECAY
                reward = parameter.STAY_REWARD * self.decay
                ticker_done = self.env.ticker_is_done()

            return obs, reward, ticker_done, info
        
    def test(self, max_episode_num, timestamp):

        test_timestamp = timestamp
        print("Test Start")

        for ep in range(int(max_episode_num)):
            # Info List - Graph 그릴 때 필요한 자료들
            info_list = []
            # 에피소드 리셋
            time, episode_reward, ticker_done = 0, 0, False
            # Env 첫 state 가져오기 & 변수 초기화
            _, state = self.env.reset(test=True)
            lowest_budget = self.env.budget
            highest_budget = self.env.budget
            # 테스트 할 곳까지 curr 옮기기
            self.env.set_curr_to_timestamp(test_timestamp)
            # Test 시작
            while not ticker_done:
                act = self.TradeAgent.get_action(state)
                long_act = np.argmax(self.LongAgent.get_action(state))
                short_act = np.argmax(self.ShortAgent.get_action(state))
                raw_act = act
                act = np.argmax(act)
                # 만약 테스트할 때 일정 확률 이상일 때만 실행시키면?? (수수료 Issue)
                if raw_act[act] < parameter.THRESHOLD:
                    if DEBUG:
                        print("pass")
                    _, state = self.env.get_next_row_obs()
                    if self.env.ticker_is_done():
                        break
                    continue
                # Action Step 실행
                next_state, reward, ticker_done, info = self.test_action_step(act)
                # 최소, 최대 Budget 갱신
                if lowest_budget > self.env.budget:
                    lowest_budget = self.env.budget
                if highest_budget < self.env.budget:
                    highest_budget = self.env.budget
                # 거래가 이루어졌을 때 Info에 추가
                if act < 2:
                    info_list.append(info)
                else:
                    # budget_list.append(info[3])
                    pass
                # Test Ticker가 끝났을 때
                if ticker_done == True:
                    break
                # reward가 None(정상적으로 Action Step이 이루어지지 않았을 때)
                if reward is None:
                    _, state = self.env.get_next_row_obs()
                    continue
                # 디버깅용
                if act < 2 and DEBUG:
                    print("Time: ",time, " Raw_Act: ",raw_act ," Act: ",act," Reward: ",reward*100,
                          " Open: ",info[0]," Close: ",info[1], " Curr: ",info[2]," Budget: ",self.env.budget)
                # State 갱신
                state = next_state
                # reward None으로 인해 ticker가 끝에 다다를 수 도 있다.
                if ticker_done == True:
                    break
            # Test 결과 저장하기
            test_reward_txt = os.path.join(BASE_DIR, "save_weights/test_reward.txt")
            with open(test_reward_txt, 'a') as f:
                f.write(f"Ep{ep} ticker: {tickers[self.env.curr_ticker]}, Budget: {self.env.budget},"+
                         " LowB: {lowest_budget}, HigiB: {highest_budget}\n")
            # 거래 info 저장하기
            df = pd.DataFrame(info_list, columns=['open', 'close', 'end', 'start', 'budget', 'percent','position'])
            print(df)
            df.to_csv(f"{BASE_DIR}/save_weights/{tickers[self.env.curr_ticker]}_test_result.csv")

        print("Test Finished")

    def test_action_step(self, action):
        ticker_done = False
        info = ["None", "None", self.env.curr]
        start_curr = self.env.curr
        # action이 매수(0, 1)이면
        if action == 0 or action == 1:
            ohlcv, state = self.env.get_next_row_obs()
            # state가 존재하지 않을 때
            if state is None:
                # ticker가 끝났는가?
                if self.env.ticker_is_done():
                    ticker_done = True # ticker끝
                else:
                    ticker_done = False
                info = ["Obs is none"]
                return None, None, ticker_done, info # None, None, True, _
            position = ohlcv['open']
            close_position = None
            # 포지션이 종료될 때 까지 action
            while(1):
                # Action 뽑아내기
                if action == 0:
                    act = self.LongAgent.get_action(state)
                else:
                    act = self.ShortAgent.get_action(state)
                act = np.argmax(act)
                # Limit: 나는 종료하고 싶은데, trade Agent가 Long을 유지하면 어차피 다시 사야함! 
                # 그래서 Trade Agent의 눈치를 본다...
                if parameter.TASIGNALON:
                    tact = self.TradeAgent.get_action(state)
                    trade_act = np.argmax(tact)
                    if action == 0:
                        if act == 1 and trade_act == 0 :
                            act = 0
                    if action == 1:
                        if act == 1 and trade_act == 1 :
                            act = 0
                # Action 분기
                if action == 0:
                    is_short = False
                else:
                    is_short = True
                # Step 밟기
                next_state, _, step_done, finish_info = self.env.long_or_short_step(act, position, is_short)
                # Step이 정상적으로 이루어지지 않았을 때
                if next_state is None:
                    # ticker가 끝났는가?
                    if self.env.ticker_is_done():
                        ticker_done = True # ticker끝
                    else:
                        ticker_done = False
                    info = ["Obs is none"]
                    return None, None, ticker_done, info # None, None, True, _
                # State 갱신
                state = next_state
                # Step이 끝났을 때
                if step_done:
                    close_position = finish_info[0]
                    # 따로 reward를 계산해줘야 한다.
                    percent = self.env.cal_percent(position, close_position)
                    if action==1:
                        percent = -percent
                    if action == 0 :
                        percent = percent - parameter.LONG_TRANS_FEE * 2
                    elif action ==1:
                        percent = percent - parameter.SHORT_TRANS_FEE * 2
                    # Budget update
                    self.env.budget *= (1+percent)
                    break # 거래가 끝났으므로 빠져나간다.
            # Info 구축하기
            obs = state
            info = [position, close_position, self.env.curr]
            ### 데이터 추가
            # 모델의 거래별 홀딩 시간
            # hold_time = end_curr - start_curr
            info.append(start_curr)
            # 모델의 budget
            info.append(self.env.budget)
            # 모델의 거래 손익 %
            info.append(percent * 100)
            # 모델의 포지션 ( Long, Short )
            info.append(action)
            return obs, percent, ticker_done, info
        # 관망(2)이면 그냥 다음행 주기
        else:
            _, obs = self.env.get_next_row_obs()
            ticker_done = self.env.ticker_is_done()
            info = ["STAY", "HOLD", self.env.curr]
            # 모델의 budget
            info.append(self.env.budget)
            # 모델의 거래 손익 %
            return obs, 0, ticker_done, info
        
def test():
    agent = Train(time_steps=0, agent_type=ALL_TYPE)
    # agent.load_weights("save_weights/a2c_17")
    # agent.test(len(tickers))
    
def train():
    max_episode_num = 200
    agent = Train(time_steps=0, agent_type=ALL_TYPE, tickers=parameter.tickers, budget=10000)
    # agent.load_weights("save_weights/a2c_19_tmp")
    agent.train(max_episode_num, 0)

if __name__=='__main__':
    train()