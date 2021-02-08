import numpy as np
from scipy.interpolate import interp1d
from math import pi
from functools import partial




class QueryClass:
    @staticmethod
    def make_from(classes_params):
        return { key: QueryClass(classes_params[key]) for key in classes_params }
    def __init__(self, params):
        self.params = params
    def sample(self, T, dt_command, is_pwm_1):
        """
    Generates a query

    Creates a query corresponding to the class of queries you are looking for
    Can be easy, middle or hard. The time duration and the amplitude can change, for now we have: 
    Easy: duration between 0.8 and 1 and amplitude between -0.3 and 0.3
    Middle: duration between 0.3 and 0.5 and amplitude between -0.4 and 0.4
    Hard: duration between 0.1 and 0.3 and amplitude between -0.6 and 0.6

    @param query_class Can be easy, middle or hard
        """
        
        breaking_points_times = [dt_command*i for i in range(int(T//dt_command))] 
         # the integral of the query in order to give a query that prevents from having big angles
        duration_min = self.params['duration'][0]
        amplitude_min = self.params['amplitude'][0]
        duration_max = self.params['duration'][1]
        amplitude_max = self.params['amplitude'][1]
        max_diff = self.params["max_diff"]
        max_angle = self.params["max_angle"]*pi/180
        if self.params["distribution"] == "uniform":
            self.distribution = np.random.uniform
            #self.distribution = partial(distribution, param1=0.1, param2=-0.3, param3 = 0.5)
        f_query = []
        for i in range(3):
            query_amplitude = self.distribution(amplitude_min,amplitude_max)
            list_query_amplitudes = [query_amplitude] # all the amplitudes sampled from the distribution using rejection sampling
            first_query_times = [0] # the list of times which are the sum of all the previous query_durations
            int_query = 0
            time = 0
            previous_amplitude = query_amplitude
            while time < T:
                query_duration = self.distribution(duration_min,duration_max)
                query_amplitude = self.distribution(amplitude_min,amplitude_max)
                new_int_query = int_query + query_duration*query_amplitude
                while (abs(query_amplitude - previous_amplitude) > max_diff) or (abs(new_int_query)>max_angle):
                    query_amplitude = self.distribution(amplitude_min,amplitude_max)
                    query_duration = self.distribution(duration_min,duration_max)
                    new_int_query = int_query + query_duration*query_amplitude
                time += query_duration
                first_query_times.append(time)
                list_query_amplitudes.append(query_amplitude)
                previous_amplitude = query_amplitude
                int_query = new_int_query
            if (i == 2) and is_pwm_1:
                list_query_amplitudes = np.zeros(len(list_query_amplitudes))
            f_query_1D = interp1d(first_query_times, list_query_amplitudes, kind="previous") # interpolation of the list first_query_times and the list list_query_amplitudes
            y = np.array(list(map(f_query_1D, breaking_points_times))) # maps the times and the new function to have a new array containing all the amplitudes corresponding to the breaking points
            f_query_1D = interp1d(np.append(breaking_points_times,[T+1]),np.append(y,[0]), kind="previous") #interpolation of the previoius array and the breaking_points_times one
            f_query.append(f_query_1D)
        return f_query
