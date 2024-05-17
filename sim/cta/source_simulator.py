#
#
#

from time import time
START=time()
import argparse
import astropy.units as u
import logging
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

import traceback
import warnings

from astropy.coordinates import Angle, SkyCoord
from astropy.utils.exceptions import AstropyWarning
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from matplotlib import use
from pathlib import Path
from regions import CircleSkyRegion, PointSkyRegion

from gammapy.data import Observation, observatory_locations, FixedPointingInfo, PointingMode
from gammapy.datasets import MapDataset, MapDatasetEventSampler
from gammapy.extern import xmltodict
from gammapy.irf import load_irf_dict_from_file
from gammapy.makers import MapDatasetMaker
from gammapy.maps import MapAxis, RegionNDMap, WcsGeom
from gammapy.modeling.models import (
    ConstantSpectralModel,
    FoVBackgroundModel,
    LightCurveTemplateTemporalModel,
    PointSpatialModel,
    PowerLawSpectralModel,
    LightCurveTemplateTemporalModel,
    ExpDecayTemporalModel,
    SkyModel,
    Models
)
from gammapy.utils.time import time_ref_to_dict


######################################
# Logging and Warnings
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(relativeCreated)6d %(process)d/%(thread)s %(name)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning, append=True)
warnings.simplefilter('ignore', category=AstropyWarning)
use("Agg")
######################################





###########

def make_obs(obsid, pointing, irf, time_ref, location="cta_south", livetime=1200*u.s):
    # TODO: separate observation time ref and start to express times in UNIX
    location = observatory_locations[location]
    pointing = FixedPointingInfo(mode=PointingMode.POINTING, fixed_icrs=pointing)
    observation = Observation.create(
        obs_id=obsid,
        pointing=pointing,
        livetime=livetime,
        irfs=irf,
        location=location,
        reference_time=time_ref,
        )
    
    return observation

def make_dataset(pointing, observation,
                 energy_axis, energy_axis_true, migra_axis,
                 width=12*u.deg, binsz=0.01*u.deg):
    """
    Define the dataset object.

    Input:
        pointing: ~astropy.SkyCoord object
        observation: ~gammapy.Observation

    Output:
        dataset: ~gammapy.Dataset
        geom: ~gammapy.maps, geometry of the dataset
    """
    geom = WcsGeom.create(skydir=pointing,
                          width=(width, width),
                          binsz=binsz,
                          frame="icrs",
                          axes=[energy_axis],
                          )
    empty = MapDataset.create(geom,
                              energy_axis_true=energy_axis_true,
                              migra_axis=migra_axis,
                              name="my-dataset"
                              )
    maker = MapDatasetMaker(selection=["exposure", "background", "psf", "edisp"])
    dataset = maker.run(empty, observation)

    return dataset

def make_default_model(time_ref, livetime, timedelay, source_coordinates, i, output_directory,
                       amplitude="3e-10 cm-2 s-1 TeV-1", #"1e-12 cm-2 s-1 TeV-1",
                       index="2.25"
                       ):
    """
    Make a default SkyModel.
    
    time_ref : `astropy.time.Time`
        Observation start.
    livetime : `astropy.quantity.Quantity`
        Observation duration.
    timedelay : `astropy.quantity.Quantity`
        Time delay of transient default model.
    source_coordinates : `astropy.coordinates.SkyCoord`
        Coordinates for Point-like source.
    i : int
        Number of the Simulation.
    output_directory : `pathlib.Path`
        Directory where to save models.
    amplitude, index : str
        Power Law parameters.
    """
    # The Default Temporal Model is an ExpDecayTemporalModel,
    # ExpDecayTemporalModel = exp((t-tstart)/t0).
    
    # Burst Offset wrt observation start.
    tstart = time_ref + timedelay
    
    # Decay Timescale t0 randomly chosen between 0 and half the livetime duration.
    t0 = np.random.uniform(0, livetime.value * 0.5) * u.s
    logger.info(f"Decay Time Scale: {t0.value:.5f} s")
    
    # The Normalization is a random factor between 0.1 and 3
    mult_fact = np.random.uniform(0.1, 3)
    logger.info(f"Normalization Factor: {mult_fact:.3f}")

    # Create the time array and define the model
    time = time_ref + np.linspace(0, livetime.to("s").value, 10000) * u.s
    idx = np.where(time >= tstart)
    expdecay_model = ExpDecayTemporalModel(t_ref=(tstart-time_ref).to("d"), t0=t0)
    # Create the norm array and evaluate the model.
    norm = np.zeros_like(time, float)
    norm[idx] = expdecay_model.evaluate(time[idx], t0, tstart).value * mult_fact
    # Write the Model
    tab = Table()
    tab["TIME"] = (time - time_ref).to("s")
    tab["NORM"] = norm
    tab.meta = time_ref_to_dict(time_ref, scale="utc")
    tab.meta["TIMEUNIT"] = "s"
    filename = output_directory.joinpath(f"lightcurve_model_{i}.fits")
    tab.write(filename, overwrite=True)
    
    # Save model plot
    fig, ax = plt.subplots(1, figsize=(12,8), constrained_layout=True)
    ax.plot(tab['TIME'], tab['NORM'])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Norm')
    ax.grid()
    fig.savefig(output_directory.joinpath(f"lightcurve_model_{i}.png"))

    # Read the Default Temporal Model produced
    temporal_model = LightCurveTemplateTemporalModel.read(filename)
    # Add a Power Law Spectral Model
    spectral_model = PowerLawSpectralModel(amplitude=amplitude, index = index, reference="1 TeV")
    # Add a Spatial Model at the source_coordinates
    spatial_model = PointSpatialModel.from_position(source_coordinates)
    # Bundle the Source Models
    model = SkyModel(spectral_model=spectral_model,
                    spatial_model=spatial_model,
                    temporal_model=temporal_model,
                    name="fake_src")
    # Add the Background
    bkg = FoVBackgroundModel(dataset_name="my-dataset")
    
    # Bundle all the Models
    models = Models([model, bkg])
    models.write(output_directory.joinpath(f"models_{i}.yaml"))
    
    return models


def save_events(events, dataset, output_directory, index):
    """Save simulated event list.

    Input:
        events: ~gammapy.EventList; event list
        dataset: ~gammapy.Dataset; dataset of the OB
        output_directory: ~path; path of the output file
        index : int; simulation index.
    """
    # Save plot
    gs = gridspec.GridSpec(nrows=2, ncols=3)
    fig = plt.figure(figsize=(12, 8))

    # energy plot
    ax_energy = fig.add_subplot(gs[1, 0])
    events.plot_energy(ax=ax_energy)

    # offset plots
    ax_offset = fig.add_subplot(gs[0, 1])
    events.plot_offset2_distribution(ax=ax_offset)
    ax_energy_offset = fig.add_subplot(gs[0, 2])
    events.plot_energy_offset(ax=ax_energy_offset)

    # time plot
    ax_time = fig.add_subplot(gs[1, 1])
    events.plot_time(ax=ax_time)

    # image plot
    m = events._counts_image(allsky=False)
    ax_image = fig.add_subplot(gs[0, 0], projection=m.geom.wcs)
    m.plot(ax=ax_image, stretch="sqrt", vmin=0)
    plt.subplots_adjust(wspace=0.3)
    
    fig.savefig(output_directory.joinpath(f"events_{index}.png"))
    
    
    # Save file
    primary_hdu = fits.PrimaryHDU()
    hdu_evt = fits.BinTableHDU(events.table)
    hdu_gti = dataset.gti.to_table_hdu()
    hdu_all = fits.HDUList([primary_hdu, hdu_evt, hdu_gti])
    hdu_all.writeto(output_directory.joinpath(f"events_{index}.fits"), overwrite=True)
    return None



def run_all(irf, time_ref, livetime, ra, dec, width, binsz, emin, emax, skymodel, nsim, outdir, lightcurveflag, lightcurvesteps, timedelay):
    """
    Perform Simulations.
    
    Parameters
    ----------
    irf : str
        Path to the IRF file.
    time_ref : str
        Observation Start time, in ISOT UTC.
    livetime : float
        Observation Live time, in seconds.
    ra, dec : float
        FoV center coordinates in deg.
    width : float
        FoV width in deg.
    binsz : float
        FoV resolution to simulate events, in deg.
    emin, emax : str
        Energy bounds for the simulated events, in astropy strings.
    skymodel : str
        Sky Model simulation. If None, a default is used.
    nsim : int
        Number of Simulations to make.
    outdir : str
        Output Directory path.
    lightcurveflag : bool
        If True, plot and save the counts lightcurve of the first model.
    lightcurvesteps : list
        Bins of the lightcurve.
    timedelay : float
        Time delay of transient default model
    """
    
    # 1 - SETUP
    logger.info(f"=====Setting initial configuration...")
    
    logger.info(f"Read IRF from: {irf}")
    irf = load_irf_dict_from_file(irf)
    
    energy_axis      = MapAxis.from_energy_bounds(emin, emax, nbin=5, per_decade=True)
    energy_axis_true = MapAxis.from_energy_bounds("0.001 TeV", "250 TeV", nbin=10, per_decade=True, name="energy_true")
    migra_axis = MapAxis.from_bounds(0.5, 2, nbin=150, node_type="edges", name="migra")
    
    pointing = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs", unit="deg")
    
    livetime = livetime * u.s
    timedelay= timedelay* u.s
    time_ref = Time(time_ref, format="isot", scale="utc")
    
    obsid = "0001" #TODO: Do not make hardcoded?
    
    output_directory = Path(outdir).absolute()
    output_directory.mkdir(parents=True, exist_ok=True)

    logger.info(f"Setting initial configuration... done!\n")
    
    # 2 - CREATE OBSERVATION and DATASET objects
    logger.info(f"=====Create observation...")
    observation = make_obs(obsid=obsid, pointing=pointing, irf=irf, time_ref=time_ref, livetime=livetime)
    logger.info(f"Create observation... done!\n")

    logger.info(f"=====Create dataset...")
    dataset = make_dataset(pointing, observation, energy_axis, energy_axis_true, migra_axis, width=width, binsz=binsz)
    logger.info(f"Create dataset... done!\n")

    # 3 - RUN SIMULATION
    logger.info(f"=====Run {nsim} simulations... ")
    for i in np.arange(nsim):
        
        # Define Models
        if skymodel is None:
            logger.info(f"Create the model and assign to the dataset...")
            source_coordinates = SkyCoord((ra+0.4) * u.deg, dec * u.deg, frame="icrs", unit="deg")
            models = make_default_model(time_ref, livetime, timedelay, source_coordinates, i, output_directory)
        else:
            logger.info(f"Read models from: {skymodel}")
            models = Models.read(skymodel)
            # Add background
            logger.info(f"Add background: {skymodel}")
            bkg = FoVBackgroundModel(dataset_name="my-dataset")
            models.append(bkg)
        logger.info(models)
        dataset.models = models
        logger.info(f"Create the model and assign to the dataset... done!")

        logger.info(f"Run simulations for pointing {i}... ")
        sampler = MapDatasetEventSampler(random_state=i)
        events = sampler.run(dataset, observation)

        logger.info(f"     - Saving events for pointing {obsid}")
        save_events(events, dataset, output_directory, i)
        logger.info(f"Run simulations for pointing {i}... done!")
        
        if lightcurveflag:
            # Get the counts light curve
            logger.info(f"Make light curves... ")
            ON_center=models[0].spatial_model.position
            ON_region = CircleSkyRegion(center=ON_center, radius=Angle(0.2*u.deg))
            ON_events = events.select_region(ON_region)

            # Compute the OFF position with background only
            # OFF has the same offset as ON position, but is symmetrical wrt pointing.
            pos_angle = pointing.position_angle(ON_center)
            sep_angle = pointing.separation(ON_center)
            OFF_center = pointing.directional_offset_by(pos_angle + Angle(np.pi, "rad"), sep_angle)
            OFF_region = CircleSkyRegion(center=OFF_center, radius=Angle(0.2*u.deg))
            OFF_events = events.select_region(OFF_region)

            # Bin Counts
            event_times_on = ON_events.time.unix
            event_times_off= OFF_events.time.unix
            t_min, t_max = np.min(event_times_on), np.max(event_times_on)
            for step in lightcurvesteps:
                n_bins = int( livetime.to('s').value/step)
                timebins = np.linspace(t_min, t_max, num=n_bins+1)

                # Bin ON and OFF light curve
                lightcurve_on, edges_on = np.histogram(event_times_on, bins=timebins)
                lightcurve_off,edges_off= np.histogram(event_times_off, bins=timebins) # edges_on==edges_off
                # Plot
                fig, ax = plt.subplots(1, figsize=(12,8), constrained_layout=True)
                ax.stairs(lightcurve_on, edges_on, label="ON")
                ax.stairs(lightcurve_off, edges_off, label="OFF")
                ax.set_xlabel('UNIX Time (s)')
                ax.set_ylabel('Counts')
                ax.set_title(f"Light Curve Simulation {i}, {step}s.")
                ax.grid()
                ax.legend()
                fig.savefig(output_directory.joinpath(f"lightcurve_simulation_{i}_{step}s.png"))
                # Save Table
                edges_min = edges_on[:-1]
                edges_max = edges_on[1:]
                t = Table([edges_min, edges_max, lightcurve_on, lightcurve_off], names=('t_min', 't_max', 'on_counts','off_counts'))
                t.write(output_directory.joinpath(f"lightcurve_simulation_{i}_{step}s.csv"))
        
    logger.info("Simulations completed!")
    return None



######################################
# Main
if __name__ == "__main__":
    # Time Monitoring
    Imports_Time = time()-START
    logger.info(f"Runtime Imports = {float(Imports_Time):.3f} s.")
    
    # Read arguments
    parser = argparse.ArgumentParser(prog='CTAOSimulator', description='Simulate a Source with CTAO IRFs', epilog="Use -h for help", formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument("--irf"     , type=str  , required=True                , help=f"Path to IRF file.")
    parser.add_argument("--time_ref", type=str  , default="2024-01-01T00:00:00", help=f"Observation start time, UTC time in ISOT format")
    parser.add_argument("--livetime", type=float, default=1200                 , help=f"Livetime simulation (in s)")
    parser.add_argument("--ra"      , type=float, default=83.63                , help=f"FoV center RA  coordinate (in deg)")
    parser.add_argument("--dec"     , type=float, default=22.01                , help=f"FoV center Dec coordinate (in deg)")
    parser.add_argument("--width"   , type=float, default=5                    , help=f"Size of the simulated FoV (in deg)")
    parser.add_argument("--binsz"   , type=float, default=0.01                 , help=f"Bin size of the simulated pixels (in deg)")
    parser.add_argument("--emin"    , type=str  , default="0.012589254 TeV"    , help=f"Energy min of the simulations, e.g. \'0.1 TeV\'")
    parser.add_argument("--emax"    , type=str  , default="199.52623 TeV"      , help=f"Energy max of the simulations, e.g. \'0.1 TeV\'")
    parser.add_argument("--skymodel", type=str  , required=False               , help=f"Path to a YAML file with model to simulate. If not provided, a default model is assumed.")
    parser.add_argument("--timedelay",type=float, default=100.0                , help=f"With a default model, delay of transient onset wrt observation start (if >0 transient starts after observation, if <0 transient starts before).")
    parser.add_argument("--nsim"    , type=int  , default=1                    , help=f"How many simulations must be performed")
    parser.add_argument("--outdir"  , type=str  , default="./events"           , help=f"Output directory.")
    parser.add_argument("--show_warnings", action="store_true"                 , help=f"If flag is set, show warnings.")
    parser.add_argument("--lightcurve",    action="store_true"                 , help=f"If flag is set, plot lightcurve of first model.")
    parser.add_argument("--lightcurvesteps", type=float, nargs='+', default=[5,10,50,100]  , help=f"Time Binning of the lightcurves to be produced in s")

    args = parser.parse_args()
        
    if not args.show_warnings:
        warnings.simplefilter("ignore")
    
    try:
        run_all(args.irf, args.time_ref, args.livetime, args.ra, args.dec, args.width, args.binsz,
                args.emin, args.emax, args.skymodel, args.nsim, args.outdir, args.lightcurve, args.lightcurvesteps, args.timedelay)
    except Exception as e:
        traceback.print_exc()
        exit(1)
    finally:
        # Time Monitoring
        logger.info(f"TOTAL RUNTIME = {float(time()-START):.3f} s.\n")
