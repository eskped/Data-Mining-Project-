
# filt_df = df.iloc[:, :-len(self.nominal_columns)]
# low = .05
# high = .95
# quant_df = filt_df.quantile([low, high])
# filt_df = filt_df.apply(lambda x: x[(x > quant_df.loc[low, x.name]) &
#                                     (x < quant_df.loc[high, x.name])], axis=0)
# filt_df = pd.concat(
#     [df.iloc[:, :-len(self.nominal_columns)], filt_df], axis=1)
# filt_df.dropna(inplace=True)
# data = filt_df.copy()
