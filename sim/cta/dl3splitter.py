from time import time
START=time()

import argparse
import logging
import numpy as np
import traceback
import warnings

from astropy.io import fits
from astropy.units import Quantity
from astropy.time import Time
from gammapy.data import EventList, Observation, GTI
from gammapy.data.metadata import ObservationMetaData
from gammapy.data.pointing import FixedPointingInfo
from gammapy.utils.deprecation import GammapyDeprecationWarning
from gammapy.utils.metadata import CreatorMetaData, TargetMetaData, TimeInfoMetaData
from gammapy.utils.scripts import make_path
from pathlib import Path
from tqdm import tqdm

# Logging
def get_logger(name, outputlogfile=None):
    """Define a simple logger"""
    
    # Stream handler for STDOUT
    StreamHandler = logging.StreamHandler()
    LogFormatter = logging.Formatter('%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')
    StreamHandler.setFormatter(LogFormatter)
    StreamHandler.setLevel(logging.INFO)
    
    # Instantiate logger
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.addHandler(StreamHandler)
    
    # Stream handler for FILE
    if outputlogfile is not None:
        FileHandler = logging.FileHandler(outputlogfile, mode='w')
        LogFormatter = logging.Formatter('%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')
        FileHandler.setFormatter(LogFormatter)
        FileHandler.setLevel(logging.DEBUG)
        log.addHandler(FileHandler)

    return log


def Observation_create(
        pointing,
        location=None,
        obs_id=0,
        livetime=None,
        tstart=None,
        tstop=None,
        irfs=None,
        deadtime_fraction=0.0,
        reference_time=Time("2000-01-01 00:00:00"),
        events=None
    ):
        """Create an observation.

        User must either provide the livetime, or the start and stop times.

        Parameters
        ----------
        pointing : `~gammapy.data.FixedPointingInfo` or `~astropy.coordinates.SkyCoord`
            Pointing information.
        location : `~astropy.coordinates.EarthLocation`, optional
            Earth location of the observatory. Default is None.
        obs_id : int, optional
            Observation ID as identifier. Default is 0.
        livetime : ~astropy.units.Quantity`, optional
            Livetime exposure of the simulated observation. Default is None.
        tstart : `~astropy.time.Time` or `~astropy.units.Quantity`, optional
            Start time of observation as `~astropy.time.Time` or duration
            relative to `reference_time`. Default is None.
        tstop : `astropy.time.Time` or `~astropy.units.Quantity`, optional
            Stop time of observation as `~astropy.time.Time` or duration
            relative to `reference_time`. Default is None.
        irfs : dict, optional
            IRFs used for simulating the observation: `bkg`, `aeff`, `psf`, `edisp`, `rad_max`. Default is None.
        deadtime_fraction : float, optional
            Deadtime fraction. Default is 0.
        reference_time : `~astropy.time.Time`, optional
            the reference time to use in GTI definition. Default is `~astropy.time.Time("2000-01-01 00:00:00")`.

        Returns
        -------
        obs : `gammapy.data.MemoryObservation`
            Observation.
        """
        if tstart is None:
            tstart = reference_time.copy()

        if tstop is None:
            tstop = tstart + Quantity(livetime)

        gti = GTI.create(tstart, tstop, reference_time=reference_time)
        obs_info = Observation._get_obs_info(
            pointing=pointing,
            deadtime_fraction=deadtime_fraction,
            time_start=gti.time_start[0],
            time_stop=gti.time_stop[0],
            reference_time=reference_time,
            location=location,
        )

        time_info = TimeInfoMetaData(
            time_start=gti.time_start[0],
            time_stop=gti.time_stop[-1],
            reference_time=reference_time,
        )

        meta = ObservationMetaData(
            deadtime_fraction=deadtime_fraction,
            location=location,
            time_info=time_info,
            creation=CreatorMetaData(),
            target=TargetMetaData(),
        )

        if not isinstance(pointing, FixedPointingInfo):
            # warnings.warn(
            #     "Pointing will be required to be provided as FixedPointingInfo",
            #     GammapyDeprecationWarning,
            # )
            pointing = FixedPointingInfo.from_fits_header(obs_info)

        if irfs is not None:
            return Observation(
                obs_id=obs_id,
                meta=meta,
                gti=gti,
                aeff=irfs.get("aeff"),
                bkg=irfs.get("bkg"),
                edisp=irfs.get("edisp"),
                psf=irfs.get("psf"),
                rad_max=irfs.get("rad_max"),
                pointing=pointing,
                location=location,
            )
        else:
            return Observation(
                obs_id=obs_id,
                meta=meta,
                gti=gti,
                aeff=None,
                bkg=None,
                edisp=None,
                psf=None,
                rad_max=None,
                pointing=pointing,
                location=location,
                events=events
            )


def Observation_write(observation, path, overwrite=False, format="gadf", include_irfs=True, checksum=False):
    """
    Write this observation into `~pathlib.Path` using the specified format.

    Parameters
    ----------
    path : str or `~pathlib.Path`
        Path for the output file.
    overwrite : bool, optional
        Overwrite existing file. Default is False.
    format : {"gadf"}
        Output format, currently only "gadf" is supported. Default is "gadf".
    include_irfs : bool, optional
        Whether to include irf components in the output file. Default is True.
    checksum : bool, optional
        When True adds both DATASUM and CHECKSUM cards to the headers written to the file.
        Default is False.
    """
    if format != "gadf":
        raise ValueError(f'Only the "gadf" format is supported, got {format}')

    path = make_path(path)

    primary = fits.PrimaryHDU()

    primary.header.update(observation.meta.creation.to_header(format))

    hdul = fits.HDUList([primary])

    events = observation.events
    if events is not None:
        events_hdu = events.to_table_hdu(format=format)
        events_hdu.header.update(observation.pointing.to_fits_header())
        
        ###################################
        # METADATA BUG FIX
        hdr = events_hdu.header
        hdr['TSTART']  = observation.events.table['TIME'].min()
        hdr['TSTOP' ]  = observation.events.table['TIME'].max()
        hdr['ONTIME']  = hdr['TSTOP' ]-hdr['TSTART']
        hdr['LIVETIME']= hdr['ONTIME']*hdr['DEADC']
        hdr['TELAPSE'] = hdr['ONTIME']
        hdr['MJDREFI'] = np.floor(observation.gti.time_ref.mjd)
        hdr['MJDREFF'] = observation.gti.time_ref.mjd - hdr['MJDREFI']
        hdr['TIMESYS'] = observation.gti.time_ref.scale
        ###################################
        
        hdul.append(events_hdu)
        
    

    gti = observation.gti
    if gti is not None:
        hdul.append(gti.to_table_hdu(format=format))

    if include_irfs:
        for irf_name in observation.available_irfs:
            irf = getattr(observation, irf_name)
            if irf is not None:
                hdul.append(irf.to_table_hdu(format="gadf-dl3"))

    hdul.writeto(path, overwrite=overwrite, checksum=checksum)
        
    return None






class DL3Splitter():
    
    def __init__(self, file : Path, stepsize : float, nlines : int, outdir : Path, seed, template, log):
        """
        Initialize.
        
        Parameters
        ----------
        file : `pathlib.Path`
            Path to the DL3 file.
        stepsize : float
            Typical size of a step in s.
        nlines : int
            Number of parallel lines.
        outdir : `pathlib.Path`
            Path to output directory.
        seed : int
            Random generator seed.
        template : str
            DL3 output file template.
        log : `logging.Logger`
            Logger.
        """
        self.file = file
        self.stepsize = stepsize
        self.nlines = nlines
        self.outdir = outdir
        self.seed = seed
        self.template = template
        self.log = log
        return None
    
    def get_dl3_name(self, name, params):
        """
        Compute name of the DL3 file.
        
        Parameters
        ----------
        name : str
            Template name for the DL3 file.
        params : dict
            Parameters of the file to replace the placeholders.
        """
        
        # "dl3_v06_sb_id_#@sb_id@#_obs_id_#@obs_id@#_tel_id_#@tel_id@#_line_idx_#@processIndex@#_thread_idx_#@threadIndex@#_th_file_idx_#@processFileIndex@#_file_idx_#@fileIndex@#.fits"
        try:
            name = name.replace("#@sb_id@#", f"{params['sb_id']}")
        except KeyError:
            pass
        try:
            name = name.replace("#@obs_id@#", f"{params['obs_id']}")
        except KeyError:
            pass
        try:
            name = name.replace("#@tel_id@#", f"{params['tel_id']}")
        except KeyError:
            pass
        try:
            name = name.replace("#@processIndex@#", f"{params['processIndex']}")
        except KeyError:
            pass
        try:
            name = name.replace("#@threadIndex@#", f"{params['threadIndex']}")
        except KeyError:
            pass
        try:
            name = name.replace("#@processFileIndex@#", f"{params['processFileIndex']}")
        except KeyError:
            pass
        try:
            name = name.replace("#@fileIndex@#", f"{params['fileIndex']}")
        except KeyError:
            pass

        return name
    
    
    def run(self):
        """
        Read a DL3 .fits file, split its events into different lines and steps, and write the files.
        """
    
        # Read the Event list
        self.log.info(f"Read {self.file}")
        events = EventList.read(self.file)

        N_events = len(events.table)
        self.log.info(f"Events: {N_events}...")

        # Assign to every event a line ID randomly
        rng = np.random.default_rng(seed=self.seed)
        assigned_line = rng.integers(low=0, high=self.nlines, endpoint=False, size=N_events)

        # Prepare time bins for the steps
        T0=events.table['TIME'].min()
        min_time = events.table['TIME'].min()-T0
        max_time = events.table['TIME'].max()-T0
        steps = int(np.ceil((max_time-min_time)/self.stepsize))
        time_edges = np.linspace(start=min_time, stop=max_time, endpoint=True, num=steps)
        self.log.info(f"T0={T0}. Min time={min_time}, Max time={max_time}. Step size={self.stepsize}s, steps={len(time_edges)-1}")
        self.log.debug(time_edges)


        # Split table into parallel lines
        N_events_all=[]
        file_counter=-1
        for i in tqdm(range(self.nlines)):

            # For every line, select the events belonging to the assigned line
            line_mask = assigned_line==i
            line_events = events.select_row_subset(line_mask)

            line_times = line_events.table['TIME']-T0
            line_N_events = len(line_events.table)
            self.log.info(f"Line {i}, {line_N_events} selected")

            # Change up the steps' time edges because steps of different lines are not synchronized
            desync = rng.normal(loc=0.0, scale=0.5, size=len(time_edges))
            line_edges= time_edges + desync
            # Make sure first step and last step include all events
            line_edges[ 0] = np.minimum(min_time, line_edges[ 0])-1.0
            line_edges[-1] = np.maximum(max_time, line_edges[-1])+1.0
            self.log.debug(line_edges)

            # Split event list into consecutive steps
            N_events_file=[]
            for j in range(len(line_edges)-1):
                file_counter+=1

                step_edge_min = line_edges[j]
                step_edge_max = line_edges[j+1]
                step_mask = (step_edge_min<line_times)&(line_times<step_edge_max)

                step_events = line_events.select_row_subset(step_mask)
                step_N_events = len(step_events.table)
                self.log.debug(f"Line {i}, step {j}, [{step_edge_min}, {step_edge_max}], {step_N_events} events selected")
                N_events_file.append(step_N_events)
                
                # Compute the file name
                params = {
                    #'sb_id':,
                    #'obs_id':,
                    #'tel_id':,
                    'processIndex':i,
                    #'threadIndex':,
                    'processFileIndex':j,
                    'fileIndex':file_counter,
                    }
                outputname = self.get_dl3_name(self.template, params)
                self.log.info(f"Write {outputname}")
                self.save_dl3_file(step_events, self.outdir.joinpath(outputname))
        
            # Sanity check: all events of the line have been distributed into step files
            self.log.info(f"Line {i}, {line_N_events} events, {np.sum(N_events_file)} selected")
            assert line_N_events==np.sum(N_events_file)
            N_events_all.append(line_N_events)

        # Sanity check: all events have been distributed into files
        self.log.info(f"Total {N_events} events, {np.sum(N_events_all)} selected")
        assert N_events==np.sum(N_events_all)

        return None
    

    def save_dl3_file(self, event_list : EventList, output_path : Path):
        """
        Convert the DL3 step EventList into an Observation, then save it.
        """


        # Timing properties
        tstart= event_list.time.min()
        tstop = event_list.time.max()
        reference_time=event_list.time_ref
        deadtime_fraction=event_list.observation_dead_time_fraction

        pointing = event_list.pointing_radec
        location=event_list.observatory_earth_location
        obs_id=event_list.table.meta['OBS_ID']

        # Create Observation
        observation = Observation_create(pointing=pointing,
                                         location=location,
                                         obs_id=obs_id,
                                         tstart=tstart,
                                         tstop=tstop,
                                         irfs=None,
                                         deadtime_fraction=deadtime_fraction,
                                         reference_time=reference_time,
                                         events=event_list
                                         )

        # Write
        Observation_write(observation, output_path, overwrite=True, include_irfs=False)

        return None






def main():
    """Run the DL3 splitter"""
    
    # Time Monitoring
    Imports_Time = time()-START
    
    # Read arguments
    parser = argparse.ArgumentParser(prog='DL3 Splitter', description='Split a DL3 into several steps and lines.', epilog="Use -h for help", formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument("--file"    , type=str  , required=True        , help=f"Path to DL3 input file.")
    parser.add_argument("--stepsize", type=float, default=10.0         , help=f"Step typical duration in s.")
    parser.add_argument("--nlines"  , type=int  , default=4            , help=f"Number of reconstruction lines.")
    parser.add_argument("--outdir"  , type=str  , default="./dl3stream", help=f"Output directory.")
    parser.add_argument("--seed"    , type=int  , default=None         , help=f"Seed for random generation.")
    parser.add_argument("--template", type=str  , required=True        , help=f"Template name for the DL3 input file.")

    args = parser.parse_args()
    
    try:
        outdir = Path(args.outdir).absolute()
        outdir.mkdir(parents=True, exist_ok=True)
        
        log = get_logger("DL3 Splitter", outputlogfile=outdir.joinpath("dl3splitter.log"))
        log.info(f"Runtime Imports = {float(Imports_Time):.3f} s.")
        
        log.info(f"Output Directory={args.outdir}")
        
        file = Path(args.file).absolute()
        if file.is_file():
            log.info(f"Input DL3 file ={file}")
        else:
            log.info(f"File not Found ={file}")
            raise FileNotFoundError
        
        log.info(f"Step size={args.stepsize}s")
        log.info(f"N lines={args.nlines}")
        log.info(f"DL3 template={args.template}")
        
        if args.seed is None:
            log.warning(f"Random seed not set.")
        else:
            log.info(f"Random seed={args.seed}")
            
        dl3_splitter = DL3Splitter(file, args.stepsize, args.nlines, outdir, args.seed, args.template, log)
        dl3_splitter.run()
                
    except Exception as e:
        traceback.print_exc()
        exit(1)
    finally:
        # Time Monitoring
        log.info(f"TOTAL RUNTIME = {float(time()-START):.3f} s.\n")
    
    return None

if __name__=="__main__":
    main()
