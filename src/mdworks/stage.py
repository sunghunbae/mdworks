from rich.console import Console
from rich.table import Table


class StageSetting:
    def __init__(self):
        self.settings = []
    
    def add_stage(self, **params):
        """Add a new stage setting inheriting from the previous stage."""
        if self.settings:
            num_stages = len(self.settings)
            prev_state = self.settings[-1]
            curr_state = prev_state.copy()
            curr_state.update(params)
            for k in ['temperature', 'posres_k']:
                curr_state[f'_delta_{k}'] = curr_state[k] - prev_state[k]
            for k in ['ps',]:
                curr_state[f'_total_{k}'] = curr_state[k] + sum([self.settings[i][k] for i in range(num_stages)])
        else:
            params.update({
                'temperature' : 10.0,
                'friction' : 1.0,
                'frequency': None, # Barostat frequency
                'posres_k' : 1000.0,
                'hmr' : False,
                'ps' : 100.0,
                'fs' : 1.0,
                '_delta_temperature' : 0.0, # ramp
                '_delta_posres_k' : 0.0, # ramp
                '_total_ps' : 100.0,
                })
            curr_state = params
        self.settings.append(curr_state)
    
    def __getitem__(self, index):
        return self.settings[index]
    
    def __len__(self):
        return len(self.settings)
    
    def __repr__(self):
        return f"StageSetting({self.settings})"
    
    def show(self):
        from rich.console import Console
        from rich.table import Table

        columns = {
            'temperature': 'Temperature(K)',
            'frequency': 'BarostatFrequency',
            'friction': 'Friction(1/ps)',
            'posres_k': 'Posres_k(kJ/mol/nm**2)',
            'ps': 'Time(ps)',
            'fs': 'Timestep(fs)',
            'hmr': 'HMR',
            '_total_ps': 'Elapsed(ps)',
        }
        
        table = Table(title="Stage Settings")
        table.add_column('Stage', justify="left")
        for k, v in columns.items():
            table.add_column(v, justify="right", no_wrap=False)
        for idx, stage in enumerate(self.settings):
            values = [str(idx)] + [str(stage[k]) for k,v in columns.items()]
            table.add_row(*values)
        console = Console()
        console.print(table)