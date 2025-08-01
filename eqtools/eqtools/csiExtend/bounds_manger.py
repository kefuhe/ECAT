"""
Bounds Manager Module

This module provides the BoundsManager class for managing parameter bounds
in Bayesian fault inversion processes.
"""

import numpy as np
import yaml


class BoundsManager:
    """
    Class to manage the bounds for all parameters in the inversion.
    """
    def __init__(self, inversion_instance):
        """
        Initialize the BoundsManager with the inversion instance.
        """
        self.config = inversion_instance.config
        self.mcmc_samples = inversion_instance.mcmc_samples
        self.multifaults = inversion_instance.multifaults
        self.inversion_instance = inversion_instance
        self.lb = np.ones(self.mcmc_samples) * np.nan
        self.ub = np.ones(self.mcmc_samples) * np.nan
        self.geometry_positions = inversion_instance.geometry_positions
        self.sigmas_position = inversion_instance.sigmas_position
        self.alpha_position = inversion_instance.alpha_position
        self.slip_positions = inversion_instance.slip_positions.copy()
        self.poly_positions = inversion_instance.poly_positions.copy() 
        # Transfer slip_positions and poly_positions to the correct MCMC sampling position.
        if self.config.slip_sampling_mode == 'rake_fixed':
            total_half = 0
            for ifault in self.inversion_instance.multifaults.faults:
                lb_slip, ub_slip = self.slip_positions[ifault.name]
                lb_poly, ub_poly = self.poly_positions[ifault.name]
                self.slip_positions[ifault.name] = [lb_slip-total_half, ub_slip-total_half -len(ifault.patch)]
                total_half += len(ifault.patch)
                self.poly_positions[ifault.name] = [lb_poly-total_half, ub_poly-total_half]
        self.bounds = {
            'geometry': None,
            'poly': None,
            'strikeslip': None,
            'dipslip': None,
            'rake_angle': None,
            'slip_magnitude': None,
            'alpha': None,
            'sigmas': None
        }

    def set_default_bounds(self, lb=None, ub=None):
        """
        Set the default bounds for all parameters.
        """
        lb_to_use = lb
        ub_to_use = ub

        if lb_to_use is not None:
            self.lb[np.isnan(self.lb)] = lb_to_use
        if ub_to_use is not None:
            self.ub[np.isnan(self.ub)] = ub_to_use
    
    def update_bounds_for_all_mcmc_parameters(self, geometry=None, slip_magnitude=None, rake_angle=None, strikeslip=None, dipslip=None, poly=None, sigmas=None, alpha=None):
        """
        Update the bounds for all MCMC parameters.

        Parameters:
        geometry (dict): Dictionary containing the bounds for geometry parameters.
        slip_magnitude (dict): Dictionary containing the bounds for slip magnitude.
        rake_angle (dict): Dictionary containing the bounds for rake angle.
        strikeslip (dict): Dictionary containing the bounds for strike-slip component.
        dipslip (dict): Dictionary containing the bounds for dip-slip component.
        poly (dict): Dictionary containing the bounds for polynomial parameters.
        sigmas (list): List containing the bounds for sigmas.
        alpha (list): List containing the bounds for alpha.
        """
        self.update_bounds_for_all_faults(geometry, slip_magnitude, rake_angle, strikeslip, dipslip, poly)
        self.update_bounds_for_sigmas(sigmas)
        self.update_bounds_for_alpha(alpha)
    
    def update_bounds_from_config(self, config_file=None, encoding='utf-8'):
        """
        Update parameter bounds from a configuration file.
        """
        if config_file is not None:
            try:
                with open(config_file, 'r', encoding=encoding) as file:
                    self.bounds_config = yaml.safe_load(file)
            except FileNotFoundError:
                print(f"Configuration file {config_file} not found.")
                return
            except yaml.YAMLError as e:
                print(f"Error parsing configuration file: {e}")
                return
    
        lb = self.bounds_config.get('lb', None)
        ub = self.bounds_config.get('ub', None)
        if lb is not None and ub is not None and lb > ub:
            print("Global lower bound should be less than upper bound.")
            return
    
        self.set_default_bounds(lb, ub)
        geometry = self.bounds_config.get('geometry', None)
        slip_magnitude = self.bounds_config.get('slip_magnitude', None)
        rake_angle = self.bounds_config.get('rake_angle', None)
        strikeslip = self.bounds_config.get('strikeslip', None)
        dipslip = self.bounds_config.get('dipslip', None)
        poly = self.bounds_config.get('poly', None)
        sigmas = self.bounds_config.get('sigmas', None)
        alpha = self.bounds_config.get('alpha', None)
        self.update_bounds_for_all_mcmc_parameters(geometry, slip_magnitude, rake_angle, 
                                                   strikeslip, dipslip, poly, sigmas, alpha)
    
    def update_bounds_for_all_faults(self, geometry=None, slip_magnitude=None, rake_angle=None, strikeslip=None, dipslip=None, poly=None):
        """
        Update the bounds for all faults based on the slip_sampling_mode.
        Parameters slip_magnitude, rake_angle, strikeslip, dipslip are set to None by default and are required based on the mode.
        """
        for fault_name in self.config.faults:
            if fault_name == 'defaults':
                continue
            if self.config.faults[fault_name]['geometry']['update'] and geometry.get(fault_name, None) is not None:
                self.update_bounds_for_geometry(fault_name, geometry[fault_name])
            
            # Assert checks to ensure the necessary parameters are provided based on the slip_sampling_mode
            if self.config.slip_sampling_mode == "rake_fixed":
                if slip_magnitude is not None:
                    self.update_slip_bounds_based_on_mode(fault_name, slip_magnitude.get(fault_name, None))
            elif self.config.slip_sampling_mode == "ss_ds":
                if strikeslip is not None or dipslip is not None:
                    istrikeslip = strikeslip.get(fault_name, None) if strikeslip is not None else None
                    idipslip = dipslip.get(fault_name, None) if dipslip is not None else None
                    if istrikeslip is not None or idipslip is not None:
                        self.update_slip_bounds_based_on_mode(fault_name, None, None, istrikeslip, idipslip)
            elif self.config.slip_sampling_mode == "magnitude_rake":
                if slip_magnitude is not None or rake_angle is not None:
                    islip_magnitude = slip_magnitude.get(fault_name, None) if slip_magnitude is not None else None
                    irake_angle = rake_angle.get(fault_name, None) if rake_angle is not None else None
                    if islip_magnitude is not None or irake_angle is not None:
                        self.update_slip_bounds_based_on_mode(fault_name, islip_magnitude, irake_angle)
            
            if poly is not None and poly.get(fault_name, None) is not None:
                self.update_bounds_for_poly(fault_name, poly[fault_name])
    
    def update_slip_bounds_based_on_mode(self, fault_name, slip_magnitude=None, rake_angle=None, strikeslip=None, dipslip=None):
        """
        根据slip_sampling_mode更新滑动参数的边界。
        """
        slip_start, slip_end = self.slip_positions[fault_name]
        slip_half = (slip_end + slip_start) // 2

        if self.config.slip_sampling_mode == 'rake_fixed':
            if slip_magnitude is not None:
                # update self.bounds
                if self.bounds['slip_magnitude'] is None:
                    self.bounds['slip_magnitude'] = {}
                self.bounds['slip_magnitude'][fault_name] = (slip_magnitude[0], slip_magnitude[1])
                # update slip bounds
                self.lb[slip_start:slip_end] = self.bounds['slip_magnitude'][fault_name][0]
                self.ub[slip_start:slip_end] = self.bounds['slip_magnitude'][fault_name][1]
        elif self.config.slip_sampling_mode == 'magnitude_rake':
            assert slip_magnitude is not None or rake_angle is not None, "Either slip_magnitude or rake_angle must not be None in magnitude_rake mode"
            if slip_magnitude is not None:
                # update self.bounds
                if self.bounds['slip_magnitude'] is None:
                    self.bounds['slip_magnitude'] = {}
                self.bounds['slip_magnitude'][fault_name] = (slip_magnitude[0], slip_magnitude[1])
                # update slip bounds
                self.lb[slip_start:slip_half] = self.bounds['slip_magnitude'][fault_name][0]
                self.ub[slip_start:slip_half] = self.bounds['slip_magnitude'][fault_name][1]
            if rake_angle is not None:
                # update self.bounds
                if self.bounds['rake_angle'] is None:
                    self.bounds['rake_angle'] = {}
                self.bounds['rake_angle'][fault_name] = (rake_angle[0], rake_angle[1])
                # update slip bounds
                self.lb[slip_half:slip_end] = self.bounds['rake_angle'][fault_name][0]
                self.ub[slip_half:slip_end] = self.bounds['rake_angle'][fault_name][1]
        else:  # 'ss_ds' mode
            assert strikeslip is not None or dipslip is not None, "Either strikeslip or dipslip must not be None in ss_ds mode"
            if strikeslip is not None:
                # update self.bounds
                if self.bounds['strikeslip'] is None:
                    self.bounds['strikeslip'] = {}
                self.bounds['strikeslip'][fault_name] = (strikeslip[0], strikeslip[1])
                # update slip bounds
                self.lb[slip_start:slip_half] = self.bounds['strikeslip'][fault_name][0]
                self.ub[slip_start:slip_half] = self.bounds['strikeslip'][fault_name][1]
            if dipslip is not None:
                # update self.bounds
                if self.bounds['dipslip'] is None:
                    self.bounds['dipslip'] = {}
                self.bounds['dipslip'][fault_name] = (dipslip[0], dipslip[1])
                # update slip bounds
                self.lb[slip_half:slip_end] = self.bounds['dipslip'][fault_name][0]
                self.ub[slip_half:slip_end] = self.bounds['dipslip'][fault_name][1]
    
    def get_bounds_for_F_J(self):
        '''
        Get the bounds for the F_J samples mode.
        '''
        linear_sample_start = self.inversion_instance.linear_sample_start_position
        return self.lb[linear_sample_start:], self.ub[linear_sample_start:]

    def get_bounds_for_geometry_hyperparameters(self):
        '''
        Get the bounds for the geometry hyperparameters.
        '''
        linear_sample_start = self.inversion_instance.linear_sample_start_position
        return self.lb[:linear_sample_start], self.ub[:linear_sample_start]

    def get_inequality_constraints_for_rake_angle(self, rake_angle=None):
        '''
        Generate linear constraints (A, b) for the rake angle range based on strike-slip and dip-slip components.
        Satify the following constraints:
         A * x <= b, where b = 0 vector
         rake_angle: (dict) {fault_name: (lower_bound, upper_bound)}
         rake: rake: (-180, 180) or (0, 360), anti-clockwise is positive
         rake_ub - rake_lb <= 180
        '''
        if rake_angle is None:
            rake_angle = self.bounds_config['rake_angle']
        else:
            self.bounds_config['rake_angle'] = rake_angle
        
        linear_sample_start = self.inversion_instance.linear_sample_start_position
        nlinear = self.mcmc_samples - linear_sample_start
        npatch_list = [len(fault.patch) for fault in self.multifaults.faults]
        npatch = int(np.sum(npatch_list))
        A = np.zeros((2 * npatch, nlinear))
        b = np.zeros(2 * npatch)

        patch_count = 0
        for ifault in self.multifaults.faults:
            start, end = self.slip_positions[ifault.name]
            start -= linear_sample_start
            end -= linear_sample_start
            half = (start + end) // 2
            # Get the rake angle bounds
            rake_start, rake_end = rake_angle[ifault.name]
            # Generate the linear constraints
            inpatch = len(ifault.patch)
            for i in range(inpatch):
                # cross product of slip and rake,
                # x: (ss, ds), y: (cos, sin) of rake with x.cross(y) = z, z = (ss*sin(rake) - ds*cos(rake)) < 0
                A[patch_count + i, start+i] = np.sin(np.deg2rad(rake_start))
                A[patch_count + i, half+i] = -np.cos(np.deg2rad(rake_start))
                # x: (ss, ds), y: (cos, sin) of rake with x.cross(y) = z, z = (ss*sin(rake) - ds*cos(rake)) > 0
                A[patch_count + inpatch+i, start+i] = -np.sin(np.deg2rad(rake_end))
                A[patch_count + inpatch+i, half+i] =   np.cos(np.deg2rad(rake_end))
            patch_count += inpatch
        
        return A, b

    def get_equality_constraints_for_fixed_rake(self, fixed_rake):
        '''
        Generate linear equality constraints (A_eq, b_eq) for a fixed rake angle based on strike-slip and dip-slip components.
        Satisfy the following constraints:
         A_eq * x = b_eq, where b_eq = 0 vector
         fixed_rake: (dict) {fault_name: fixed_rake_angle}
         rake: fixed value
        '''
        linear_sample_start = self.inversion_instance.linear_sample_start_position
        nlinear = self.mcmc_samples - linear_sample_start
        npatch_list = [len(fault.patch) for fault in self.multifaults.faults]
        npatch = int(np.sum(npatch_list))
        A_eq = np.zeros((npatch, nlinear))
        b_eq = np.zeros(npatch)

        patch_count = 0
        for ifault in self.multifaults.faults:
            start, end = self.slip_positions[ifault.name]
            start -= linear_sample_start
            end -= linear_sample_start
            half = (start + end) // 2
            # Get the fixed rake angle
            rake = fixed_rake[ifault.name]
            # Generate the linear constraints
            inpatch = len(ifault.patch)
            for i in range(inpatch):
                # cross product of slip and rake,
                # x: (ss, ds), y: (cos, sin) of rake with x.cross(y) = z, z = (ss*sin(rake) - ds*cos(rake)) = 0
                A_eq[patch_count + i, start+i] = np.sin(np.deg2rad(rake))
                A_eq[patch_count + i, half+i] = -np.cos(np.deg2rad(rake))
            patch_count += inpatch
        
        return A_eq, b_eq

    def update_bounds_for_sigmas(self, bounds):
        if any(self.config.sigmas['update']) and bounds is not None:
            start, end = self.sigmas_position
            self.lb[start:end] = bounds[0]
            self.ub[start:end] = bounds[1]
            # update self.bounds
            self.bounds['sigmas'] = bounds
    
    def update_bounds_for_alpha(self, bounds):
        if self.config.alpha['update'] and bounds is not None:
            start, end = self.alpha_position
            self.lb[start:end] = bounds[0]
            self.ub[start:end] = bounds[1]
            # update self.bounds
            self.bounds['alpha'] = bounds
    
    def update_bounds_for_geometry(self, fault_name, bounds):
        if self.config.faults[fault_name]['geometry']['update']:
            start, end = self.config.faults[fault_name]['geometry']['sample_positions']
            self.lb[start:end] = bounds[0]
            self.ub[start:end] = bounds[1]
            # update self.bounds
            if self.bounds['geometry'] is None:
                self.bounds['geometry'] = {}
            self.bounds['geometry'][fault_name] = bounds

    def update_bounds_for_slip(self, fault_name, bounds):
        start, end = self.slip_positions[fault_name]
        self.lb[start:end] = bounds[0]
        self.ub[start:end] = bounds[1]
        half = (start + end) // 2
        # update self.bounds
        if self.config.slip_sampling_mode == 'rake_fixed':
            if self.bounds['slip_magnitude'] is None:
                self.bounds['slip_magnitude'] = {}
            self.bounds['slip_magnitude'][fault_name] = bounds
        elif self.config.slip_sampling_mode == 'magnitude_rake':
            if self.bounds['slip_magnitude'] is None:
                self.bounds['slip_magnitude'] = {}
            if self.bounds['rake_angle'] is None:
                self.bounds['rake_angle'] = {}
            self.bounds['slip_magnitude'][fault_name] = [self.lb[start:half], self.ub[start:half]]
            self.bounds['rake_angle'][fault_name] = [self.lb[half:end], self.ub[half:end]]
        else:  # 'ss_ds' mode
            if self.bounds['strikeslip'] is None:
                self.bounds['strikeslip'] = {}
            if self.bounds['dipslip'] is None:
                self.bounds['dipslip'] = {}
            self.bounds['strikeslip'][fault_name] = [self.lb[start:half], self.ub[start:half]]
            self.bounds['dipslip'][fault_name] = [self.lb[half:end], self.ub[half:end]]
    
    def update_bounds_for_poly(self, fault_name, bounds):
        start, end = self.poly_positions[fault_name]
        self.lb[start:end] = bounds[0]
        self.ub[start:end] = bounds[1]
        # update self.bounds
        if self.bounds['poly'] is None:
            self.bounds['poly'] = {}
        self.bounds['poly'][fault_name] = bounds