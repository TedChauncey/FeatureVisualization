
T = np.linspace(0,24, X.shape[0])
E = Y_pred[:,1]
E[E>median(E)] = 1
E[E<median(E)] = 0


from lifelines import KaplanMeierFitter
kmf = KaplanMeierFitter()
kmf.fit(T, event_observed=E)  # or, more succiently, kmf.fit(T, E)

figure()
kmf.survival_function_
kmf.median_
kmf.plot()


groups = Stage
ix = (groups == '1A')

kmf.fit(T[~ix], E[~ix], label='Stage 1A')
ax = kmf.plot()

kmf.fit(T[ix], E[ix], label='Stage 1B and 2')
kmf.plot(ax=ax)
