import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ta.momentum import RSIIndicator,StochasticOscillator
import warnings

#pip install pandas matplotlib numpy ta - Use this for instaiiling the libraries being used here

warnings.filterwarnings('ignore')


class AlphaStrategy:
    def __init__(self):
        self.close = self.static_data("D:/bao/close.csv")
        self.open = self.static_data("D:/bao/open.csv")
        self.high = self.static_data("D:/bao/high.csv")
        self.low = self.static_data("D:/bao/low.csv")
        self.volume = self.static_data("D:/bao/volume.csv")
        self.amount = self.static_data("D:/bao/amount.csv")
        self.daily_notional = 100

    def static_data(self, addr):
        df = pd.read_csv(addr)
        df.set_index('time', inplace=True)
        df.index = pd.to_datetime(df.index)
        return df

    @property
    def daily_ret(self):
        dates_to_process = self.close.index.date
        unique_dates = sorted(set(dates_to_process))
        daily_returns = pd.DataFrame(index=unique_dates, columns=self.close.columns)

        for date in unique_dates:
            selected_data = self.close.loc[self.close.index.date == date]  # Select the rows for the current date

            daily_return = (selected_data.iloc[-1] / selected_data.iloc[0]) - 1
            daily_returns.loc[date] = daily_return

        return daily_returns


    @property
    def freq_intraday_ret(self):
        #this is to be used when we are rebalancing portfolio throughout the day on 30 minute data
        freq_ret = self.close.pct_change()
        freq_ret.loc[freq_ret.index.time == pd.to_datetime('10:00').time()] = 0
        return freq_ret

    @property
    def equal_weighted_csi_500(self):
        # this is used to get the perfom=rmance of the equal weighted portfolio which is bougth at 10 AM and the position is exited at 3PM
        d_r = self.daily_ret.copy()
        dates_to_process = d_r.index
        unique_dates = sorted(set(dates_to_process))
        for date, row in d_r.iterrows():
            nan_cols = row[row.isnull()].index.tolist()  # Get columns with NaN values

            # If no Nan values, distribute equal weights among all columns(stocks)
            if len(nan_cols) == 0:
                num_cols = len(row)
                equal_weight = 1 / num_cols
                row[:] = equal_weight

            # If Nan values, distribute equal weights among the remaining non-Nan columns
            else:
                valid_columns = row[row.notnull()].index.tolist()
                num_valid_columns = len(valid_columns)
                equal_weight = 1 / num_valid_columns
                row[nan_cols] = 0
                row[valid_columns] = equal_weight

            d_r.loc[date] = row

        equal_weighted_profits = self.daily_notional * d_r * self.daily_ret
        total_equal_weighted_profits = equal_weighted_profits.sum(axis=1)
        total_equal_weighted_profits = total_equal_weighted_profits.cumsum()

        return total_equal_weighted_profits,equal_weighted_profits.sum(axis=1)

    def alpha_norm_time(self,alpha):
        #this function is used to normalize the alpa for any day
        alpha_sum = alpha.sum(axis=1)
        alpha = alpha.div(alpha_sum, axis=0)
        alpha_norm = alpha.copy()
        alpha = alpha.between_time('10:00', '10:00')
        alpha.index = alpha.index.date
        return alpha,alpha_norm

    def in_os(self,df,in_out):
        #this functions splits the dataset into test and out-sample
        if in_out == 'IN':
            # ins sample data
            df = df['2022-04-01':'2022-06-30']
            return df
        elif in_out == 'OS':
            # out sample data
            df = df['2022-07-01':'2022-07-31']
            return df




    def alpha_1(self, delay,in_out,freq_indicator=1):
        """
            This alpha is  o identify securities that have a large change in their closing prices over a two-day period
            but with low trading volume relative to their average daily volume (ADV).
            The idea behind this alpha factor is that large price movements on low trading volume could indicate that the
            market has not yet fully priced in the new information, and therefore, there may be an opportunity to profit from the price move.

            Parameters:
            - delay: refers to the signal delay
            - in_out: Specifies whether the data is for in-sample ('IN') or out-of-sample ('OS')
            - freq_indicator: Indicates the frequency indicator for the strategy
            #freq_indicator is by default 1 if we are rebalancing the positions throughout the day, 0 if we buy at 10 AM and hold the same position for th entire day

            """
        df_volume = self.in_os(self.volume,in_out).shift(delay)
        df_close = self.in_os(self.close,in_out).shift(delay)
        adv_12 = df_volume.rolling(window=12).mean()
        close_delta = df_close.diff(periods = 12)
        large_price_change = close_delta>0.05*df_close
        # volume_difference = (df_volume - adv_12) / adv_12

        # ranked_volume_diff = volume_difference.rank(axis=1, pct=True)
        large_price_change = large_price_change.astype(int)
        cond_2 = (df_volume<adv_12).astype(int)
        alpha = large_price_change* cond_2
        alpha = pd.DataFrame(alpha, index=df_volume.index, columns=df_volume.columns)
        d = alpha.sum(axis=1)
        print(d.sum())
        if not freq_indicator:
            return self.alpha_norm_time(alpha)
        if delay == 1:
            alpha.loc[alpha.index.time == pd.to_datetime('10:00').time()] = 0
        return self.alpha_norm_time(alpha)


    def calculate_rsi(self,df, period=15):
        """
            This function is uses to calculate the Relative Strength Index (RSI) for each column in the DataFrame.
            Parameters:
            - df: DataFrame containing close price data for securities in csi 500
            - period: The time window used for RSI calculation (default 15)
            """
        rsi_dict = {}
        for column in df.columns:
            indicator = RSIIndicator(close=df[column], window=period)
            rsi_dict[column] = indicator.rsi()
        return pd.DataFrame(rsi_dict)

    def alpha_2(self, delay,in_out,freq_indicator=1):
        """
            This alpha computes RSI for the closing prices of stocks in csi 500 data  and tries to identify oversold conditions.

            """
        df_close = self.in_os(self.close,in_out).shift(delay)
        rsi_values = self.calculate_rsi(df_close)

        oversold_threshold = 40

        oversold_stocks = rsi_values.apply(lambda x: x < oversold_threshold).astype(int)
        oversold_stocks = oversold_stocks.where(oversold_stocks == 1, 0)
        alpha = oversold_stocks
        if not freq_indicator:
            return self.alpha_norm_time(alpha)
        if delay == 1:
            alpha.loc[alpha.index.time == pd.to_datetime('10:00').time()] = 0
        return self.alpha_norm_time(alpha)


    def alpha_3(self, window,delay,in_out,freq_indicator=1):
        """
            This alpha based on if close price goes below lower band of a Bollinger Bands.

            Parameters:
            - window: Rolling window size for calculating indicators

            Returns:
            - DataFrame: Alpha represents buying signals based on the lower band of Bollinger Bands
            """
        df_close = self.in_os(self.close,in_out).shift(delay)
        rolling_mean = df_close.rolling(window=window).mean()
        rolling_std = df_close.rolling(window=window).std()

        lower_band = rolling_mean - (rolling_std * 2)

        buy_condition = df_close.lt(lower_band)

        signals = buy_condition.astype(int)
        d = signals.sum(axis=1)
        alpha = signals
        if not freq_indicator:
            return self.alpha_norm_time(alpha)
        if delay == 1:
            alpha.loc[alpha.index.time == pd.to_datetime('10:00').time()] = 0
        return self.alpha_norm_time(alpha)


    def calculate_indicators(self,df_close, df_open, window):
        """
            Calculates indicators like including upper band, lower band, lower-lower band, and average volume.

            """
        rolling_mean = df_close.rolling(window=window).mean()
        rolling_std = df_close.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 1.5)
        lower_lower_band = rolling_mean - (rolling_std * 2)

        avg_open = df_open.rolling(window=window).mean()

        return upper_band, lower_band, lower_lower_band, avg_open

    def alpha_4(self, window, delay,in_out,freq_indicator=1):
        """
           Generates buy signals based on combination of  Bollinger Bands and volume conditions. The idea is to identify stocks that if close prices are lower tan avg close price but open prices is higher than average
           it might signal that the market might have temporarily pushed the price lower than expected at the close and  there might be a short-term price reversal or a bounce-back in the price.
           """

        df_close = self.in_os(self.close,in_out).shift(delay)
        df_open = self.in_os(self.open,in_out).shift(delay)
        upper_band, lower_band, lower_lower_band, avg_open = self.calculate_indicators(df_close, df_open, window)

        # condition_1 = (df_close.lt(lower_band))
        # condition_2 = (df_close.lt(lower_lower_band))

        buy_condition = (df_close.lt(lower_band)) & (df_open.gt(avg_open))

        buy_condition_2 = (df_close.lt(lower_lower_band)) & (df_open.gt(avg_open))

        signals = pd.DataFrame(0, index=df_close.index, columns=df_close.columns)
        signals[buy_condition] = 1
        signals[buy_condition_2] = 2
        d = signals.sum(axis=1)
        print(d.sum())
        alpha = signals
        if not freq_indicator:
            return self.alpha_norm_time(alpha)
        if delay == 1:
            alpha.loc[alpha.index.time == pd.to_datetime('10:00').time()] = 0
        return self.alpha_norm_time(alpha)

    # the below simulation is used when we are buying the position at 10 AM and holding the same till close of trading day
    def simulate_alpha_hold_init(self, alpha_df):
        alpha_df = alpha_df[0]
        start = alpha_df.index.min()
        end = alpha_df.index.max()
        daily_ret =  self.daily_ret.loc[(self.daily_ret.index >= start) & (self.daily_ret.index <= end)]
        portfolio_returns = self.daily_notional * daily_ret * alpha_df[0]
        portfolio_returns.fillna(0, inplace=True)
        daily_pnl = portfolio_returns.sum(axis=1)
        cumulative_returns = daily_pnl.cumsum()

        # Calculating Sharpe ratio, Calmar ratio
        sharpe_ratio = np.sqrt(252) * (daily_pnl.mean() / daily_pnl.std())
        prev_peak = cumulative_returns.cummax()
        drawdown = (prev_peak - cumulative_returns)
        max_drawdown = max(drawdown)
        tot_profit = cumulative_returns[-1]
        calmar_ratio = tot_profit / max_drawdown

        start = cumulative_returns.index.min()
        end = cumulative_returns.index.max()

        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_returns.index, cumulative_returns, label='Cumulative Portfolio Returns', linewidth=2)


        equal_weighted_profits = self.equal_weighted_csi_500[0].loc[(self.equal_weighted_csi_500[0].index >= start) & (self.equal_weighted_csi_500[0].index <= end)]
        daily_equal_weight_pnl = self.equal_weighted_csi_500[1].loc[(self.equal_weighted_csi_500[1].index >= start) & (self.equal_weighted_csi_500[1].index <= end)]

        ew_sharpe_ratio = np.sqrt(252) * (daily_equal_weight_pnl.mean() / daily_equal_weight_pnl.std())
        prev_peak = equal_weighted_profits.cummax()
        drawdown = (prev_peak - equal_weighted_profits)
        max_drawdown = max(drawdown)
        tot_profit = equal_weighted_profits[-1]
        ew_calmar_ratio = tot_profit / max_drawdown

        plt.plot(equal_weighted_profits.index, equal_weighted_profits, label='Equal Weighted Portfolio', linewidth=2)

        plt.title('Cumulative Returns of alpha')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.grid(True)
        plt.show()

        print("Alpha Strategy Portfolio Metrics:")
        print(f"Sharpe Ratio: {sharpe_ratio}")
        print(f"Calmar Ratio: {calmar_ratio}")

        print("\nEqual Weighted Portfolio Metrics:")
        print(f"Sharpe Ratio: {ew_sharpe_ratio}")
        print(f"Calmar Ratio: {ew_calmar_ratio}")


    #the below simulation is used when we are rebalancing the porfolio every 30 minutes
    def simulate_alpha_hold_30_freq(self,alpha_df):
        alpha_df = alpha_df[1].shift(1)
        start_date = alpha_df.index.min()
        end_date = alpha_df.index.max()
        freq_intraday_ret = self.freq_intraday_ret.loc[(self.freq_intraday_ret.index >= start_date) & (self.freq_intraday_ret.index <= end_date)]
        portfolio_returns = self.daily_notional * alpha_df * freq_intraday_ret
        portfolio_returns.fillna(0, inplace=True)
        daily_pnl = portfolio_returns.groupby(portfolio_returns.index.date).sum()
        daily_pnl = daily_pnl.sum(axis=1)
        cumulative_returns = daily_pnl.cumsum()


        # Calculating Sharpe ratio, Calmar ratio
        sharpe_ratio = np.sqrt(252) * (daily_pnl.mean() / daily_pnl.std())
        prev_peak = cumulative_returns.cummax()
        drawdown = (prev_peak - cumulative_returns)
        max_drawdown = max(drawdown)
        tot_profit = cumulative_returns[-1]
        calmar_ratio = tot_profit / max_drawdown

        start_date = cumulative_returns.index.min()
        end_date = cumulative_returns.index.max()

        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_returns.index, cumulative_returns, label='Cumulative Portfolio Returns', linewidth=2)


        # equal_weighted_profits = self.equal_weighted_csi_500[0].loc[(self.equal_weighted_csi_500[0].index >= start_date) & (self.equal_weighted_csi_500[0].index <= end_date)]
        daily_equal_weight_pnl = self.equal_weighted_csi_500[1].loc[(self.equal_weighted_csi_500[1].index >= start_date) & (self.equal_weighted_csi_500[1].index <= end_date)]
        ew_sharpe_ratio = np.sqrt(252) * (daily_equal_weight_pnl.mean() / daily_equal_weight_pnl.std())
        equal_weighted_profits = daily_equal_weight_pnl.cumsum()
        prev_peak = equal_weighted_profits.cummax()
        drawdown = (prev_peak - equal_weighted_profits)
        max_drawdown = max(drawdown)
        tot_profit = equal_weighted_profits[-1]
        ew_calmar_ratio = tot_profit / max_drawdown


        plt.plot(equal_weighted_profits.index, equal_weighted_profits, label='Equal Weighted Portfolio', linewidth=2)

        plt.title('Cumulative Returns of alpha')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.grid(True)
        plt.show()


        print("Alpha Strategy Portfolio Metrics:")
        print(f"Sharpe Ratio: {sharpe_ratio}")
        print(f"Calmar Ratio: {calmar_ratio}")

        print("\nEqual Weighted Portfolio Metrics:")
        print(f"Sharpe Ratio: {ew_sharpe_ratio}")
        print(f"Calmar Ratio: {ew_calmar_ratio}")






if __name__ == "__main__":
    alpha = AlphaStrategy()
  # Set your desired delay and the alpha strategy you want to test on, the strategy performance will be printed alonwith the performance of the equal weighted portfolio. Based on which kind of simulation we want to run we can set it.
    alpha_data = alpha.alpha_4(9, 0,'IN')
  #   alpha_data = alpha.alpha_4(13, 1,'IN')
  #   alpha_data = alpha.alpha_1(0,'IN')
  #   alpha_data = alpha.alpha_2(0, 'IN')
    alpha.simulate_alpha_hold_30_freq(alpha_data)
    # alpha.simulate_alpha_hold_init(alpha_data)
    print()



