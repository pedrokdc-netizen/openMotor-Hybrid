"""Component models and transient simulation for pressure-fed hybrid motors."""

import math

from .constants import gasConstant
from .grains.bates import BatesGrain
from .motor import MotorConfig
from .nozzle import Nozzle
from .properties import EnumProperty, FloatProperty, PropertyCollection, StringProperty
from .simResult import LogChannel, SimAlert, SimAlertLevel, SimAlertType, SimulationResult


class HybridFuel(PropertyCollection):
    """Physical properties and empirical oxidizer-flux regression law."""

    def __init__(self, properties=None):
        super().__init__()
        self.props["name"] = StringProperty("Name")
        self.props["density"] = FloatProperty("Density", "kg/m^3", 1, 10000)
        self.props["regressionA"] = FloatProperty(
            "Regression Coefficient", "m/s/(kg/(m^2*s))^n", 0, 10
        )
        self.props["regressionN"] = FloatProperty("Regression Exponent", "", 0, 2)
        self.setProperties(
            {
                "name": "HTPB",
                "density": 930,
                "regressionA": 0.0003,
                "regressionN": 0.5,
            }
        )
        if properties is not None:
            self.setProperties(properties)

    def getRegressionRate(self, oxidizerFlux):
        if oxidizerFlux <= 0:
            return 0
        return self.getProperty("regressionA") * (
            oxidizerFlux ** self.getProperty("regressionN")
        )


class HybridFuelGrain(BatesGrain):
    """Single cylindrical fuel port with inhibited end faces."""

    geomName = "Cylindrical Hybrid"

    def __init__(self, properties=None):
        super().__init__()
        self.setProperties(
            {
                "diameter": 0.1,
                "length": 0.5,
                "coreDiameter": 0.04,
                "inhibitedEnds": "Both",
            }
        )
        if properties is not None:
            self.setProperties(properties)
        self.setProperty("inhibitedEnds", "Both")

    def setProperties(self, props):
        super().setProperties(props)
        if "inhibitedEnds" in self.props:
            self.setProperty("inhibitedEnds", "Both")


class Oxidizer(PropertyCollection):
    """Constant-property incompressible oxidizer used by the MVP feed model."""

    def __init__(self, properties=None):
        super().__init__()
        self.props["name"] = StringProperty("Name")
        self.props["density"] = FloatProperty("Liquid Density", "kg/m^3", 1, 5000)
        self.setProperties({"name": "Generic Liquid", "density": 750})
        if properties is not None:
            self.setProperties(properties)


class OxidizerTank(PropertyCollection):
    """Pressurized liquid tank with polytropic ullage blowdown."""

    def __init__(self, properties=None):
        super().__init__()
        self.props["volume"] = FloatProperty("Tank Volume", "m^3", 1e-8, 100)
        self.props["oxidizerMass"] = FloatProperty(
            "Initial Oxidizer Mass", "kg", 0, 1e5
        )
        self.props["initialPressure"] = FloatProperty(
            "Initial Tank Pressure", "Pa", 0, 1e9
        )
        self.props["pressurantGamma"] = FloatProperty(
            "Pressurant Polytropic Exponent", "", 0.1, 2
        )
        self.props["residualMass"] = FloatProperty(
            "Unusable Oxidizer Mass", "kg", 0, 1e5
        )
        self.setProperties(
            {
                "volume": 0.01,
                "oxidizerMass": 4,
                "initialPressure": 5e6,
                "pressurantGamma": 1.2,
                "residualMass": 0,
            }
        )
        if properties is not None:
            self.setProperties(properties)

    def getInitialUllageVolume(self, oxidizer):
        return self.getProperty("volume") - (
            self.getProperty("oxidizerMass") / oxidizer.getProperty("density")
        )

    def getPressure(self, oxidizerMass, oxidizer):
        initialUllage = self.getInitialUllageVolume(oxidizer)
        currentUllage = self.getProperty("volume") - (
            oxidizerMass / oxidizer.getProperty("density")
        )
        if initialUllage <= 0 or currentUllage <= 0:
            return 0
        return self.getProperty("initialPressure") * (
            initialUllage / currentUllage
        ) ** self.getProperty("pressurantGamma")


class FeedSystem(PropertyCollection):
    """Unregulated or regulated pressure-fed delivery."""

    def __init__(self, properties=None):
        super().__init__()
        self.props["mode"] = EnumProperty(
            "Feed Mode", ["Pressure Fed", "Regulated"]
        )
        self.props["regulatedPressure"] = FloatProperty(
            "Regulated Pressure", "Pa", 0, 1e9
        )
        self.props["regulatorMargin"] = FloatProperty(
            "Regulator Pressure Margin", "Pa", 0, 1e9
        )
        self.props["dropoutBehavior"] = EnumProperty(
            "Regulator Dropout Behavior", ["Upstream Pressure", "Shut Off"]
        )
        self.setProperties(
            {
                "mode": "Pressure Fed",
                "regulatedPressure": 4e6,
                "regulatorMargin": 1e5,
                "dropoutBehavior": "Upstream Pressure",
            }
        )
        if properties is not None:
            self.setProperties(properties)

    def getPressure(self, tankPressure):
        if self.getProperty("mode") == "Pressure Fed":
            return tankPressure
        regulated = self.getProperty("regulatedPressure")
        if tankPressure >= regulated + self.getProperty("regulatorMargin"):
            return regulated
        if self.getProperty("dropoutBehavior") == "Upstream Pressure":
            return tankPressure
        return 0


class Injector(PropertyCollection):
    """Equivalent incompressible injector orifice."""

    def __init__(self, properties=None):
        super().__init__()
        self.props["dischargeCoefficient"] = FloatProperty(
            "Discharge Coefficient", "", 0, 1
        )
        self.props["area"] = FloatProperty("Total Flow Area", "m^2", 0, 1)
        self.setProperties({"dischargeCoefficient": 0.7, "area": 2e-5})
        if properties is not None:
            self.setProperties(properties)

    def getMassFlow(self, feedPressure, chamberPressure, oxidizerDensity):
        pressureDrop = max(feedPressure - chamberPressure, 0)
        return (
            self.getProperty("dischargeCoefficient")
            * self.getProperty("area")
            * math.sqrt(2 * oxidizerDensity * pressureDrop)
        )


class HybridCombustion(PropertyCollection):
    """Constant-property chamber gas model for the MVP hybrid solver."""

    def __init__(self, properties=None):
        super().__init__()
        self.props["temperature"] = FloatProperty(
            "Combustion Temperature", "K", 1, 10000
        )
        self.props["gamma"] = FloatProperty("Specific Heat Ratio", "", 1.000001, 3)
        self.props["molarMass"] = FloatProperty(
            "Exhaust Molar Mass", "g/mol", 1e-6, 500
        )
        self.props["efficiency"] = FloatProperty(
            "Characteristic Velocity Efficiency", "", 0.01, 1
        )
        self.props["initialChamberPressure"] = FloatProperty(
            "Initial Chamber Pressure", "Pa", 0, 1e9
        )
        self.setProperties(
            {
                "temperature": 3000,
                "gamma": 1.2,
                "molarMass": 24,
                "efficiency": 0.9,
                "initialChamberPressure": 101325,
            }
        )
        if properties is not None:
            self.setProperties(properties)

    def getSpecificGasConstant(self):
        return gasConstant / self.getProperty("molarMass")

    def getCStar(self):
        gamma = self.getProperty("gamma")
        numerator = math.sqrt(
            gamma
            * self.getSpecificGasConstant()
            * self.getProperty("temperature")
        )
        denominator = gamma * math.sqrt(
            (2 / (gamma + 1)) ** ((gamma + 1) / (gamma - 1))
        )
        return (numerator / denominator) * self.getProperty("efficiency")


class HybridConfig(MotorConfig):
    """Hybrid-specific limits and integration controls."""

    def __init__(self):
        super().__init__()
        self.props["maxOxidizerFlux"] = FloatProperty(
            "Maximum Allowed Oxidizer Flux", "kg/(m^2*s)", 0, 1e5
        )
        self.props["maxSimTime"] = FloatProperty(
            "Maximum Simulation Time", "s", 0.01, 1e4
        )
        self.props["chamberDeadVolume"] = FloatProperty(
            "Additional Chamber Free Volume", "m^3", 0, 100
        )
        self.setProperties(
            {
                "maxPressure": 1.5e7,
                "maxOxidizerFlux": 1500,
                "burnoutWebThres": 2.54e-5,
                "burnoutThrustThres": 0.1,
                "timestep": 0.005,
                "ambPressure": 101325,
                "maxSimTime": 30,
                "chamberDeadVolume": 0.0002,
            }
        )


class HybridSimulationResult(SimulationResult):
    """Simulation result with hybrid state channels and compatibility channels."""

    def __init__(self, motor):
        super().__init__(motor)
        self.channels.update(
            {
                "tankPressure": LogChannel("Tank Pressure", float, "Pa"),
                "feedPressure": LogChannel("Feed Pressure", float, "Pa"),
                "oxidizerMass": LogChannel("Oxidizer Mass", float, "kg"),
                "oxidizerMassFlow": LogChannel(
                    "Oxidizer Mass Flow", float, "kg/s"
                ),
                "fuelMass": LogChannel("Fuel Mass", float, "kg"),
                "fuelMassFlow": LogChannel("Fuel Mass Flow", float, "kg/s"),
                "totalMassFlow": LogChannel("Total Mass Flow", float, "kg/s"),
                "nozzleMassFlow": LogChannel("Nozzle Mass Flow", float, "kg/s"),
                "mixtureRatio": LogChannel("Mixture Ratio", float, ""),
                "oxidizerMassFlux": LogChannel(
                    "Oxidizer Mass Flux", float, "kg/(m^2*s)"
                ),
                "chamberGasMass": LogChannel("Chamber Gas Mass", float, "kg"),
                "chamberVolume": LogChannel("Chamber Free Volume", float, "m^3"),
                "portArea": LogChannel("Port Area", float, "m^2"),
            }
        )

    def getConsumedOxidizerMass(self, index=None):
        stop = -1 if index is None else index
        return self.channels["oxidizerMass"].getPoint(0) - self.channels[
            "oxidizerMass"
        ].getPoint(stop)

    def getConsumedFuelMass(self, index=None):
        stop = -1 if index is None else index
        return self.channels["fuelMass"].getPoint(0) - self.channels[
            "fuelMass"
        ].getPoint(stop)

    def getPropellantMass(self, index=0):
        return (
            self.channels["oxidizerMass"].getPoint(index)
            + self.channels["fuelMass"].getPoint(index)
        )

    def getISP(self, index=None):
        consumed = self.getConsumedOxidizerMass(index) + self.getConsumedFuelMass(
            index
        )
        if consumed <= 0:
            return 0
        from .constants import standardGravity

        return self.getImpulse(index) / (consumed * standardGravity)

    def getIdealThrustCoefficient(self):
        pressure = self.getAveragePressure()
        ambient = self.motor.config.getProperty("ambPressure")
        return self.motor.nozzle.getIdealThrustCoeff(
            pressure,
            ambient,
            self.motor.combustion.getProperty("gamma"),
            0,
        )

    def getAdjustedThrustCoefficient(self):
        pressure = self.getAveragePressure()
        ambient = self.motor.config.getProperty("ambPressure")
        return self.motor.nozzle.getAdjustedThrustCoeff(
            pressure,
            ambient,
            self.motor.combustion.getProperty("gamma"),
            0,
        )

    def getAverageMixtureRatio(self):
        values = [
            value
            for value in self.channels["mixtureRatio"].getData()
            if value > 0
        ]
        return sum(values) / len(values) if values else 0


class HybridMotor:
    """Minimum viable pressure-fed hybrid rocket motor."""

    motorType = "hybrid"

    def __init__(self, properties=None):
        self.fuel = HybridFuel()
        self.grain = HybridFuelGrain()
        self.grains = [self.grain]
        self.oxidizer = Oxidizer()
        self.tank = OxidizerTank()
        self.feed = FeedSystem()
        self.injector = Injector()
        self.combustion = HybridCombustion()
        self.nozzle = Nozzle()
        self.config = HybridConfig()
        self.nozzle.setProperties(
            {
                "throat": 0.015,
                "exit": 0.035,
                "efficiency": 0.9,
                "divAngle": 15,
                "convAngle": 55,
                "throatLength": 0.005,
            }
        )
        if properties is not None:
            self.applyDict(properties)

    def getDict(self):
        return {
            "motorType": "hybrid",
            "fuel": {
                "material": self.fuel.getProperties(),
                "grain": {
                    "type": self.grain.geomName,
                    "properties": self.grain.getProperties(),
                },
            },
            "oxidizer": self.oxidizer.getProperties(),
            "tank": self.tank.getProperties(),
            "feed": self.feed.getProperties(),
            "injector": self.injector.getProperties(),
            "combustion": self.combustion.getProperties(),
            "nozzle": self.nozzle.getProperties(),
            "config": self.config.getProperties(),
        }

    def applyDict(self, data):
        fuel = data.get("fuel", {})
        self.fuel.setProperties(fuel.get("material", {}))
        self.grain.setProperties(fuel.get("grain", {}).get("properties", {}))
        self.oxidizer.setProperties(data.get("oxidizer", {}))
        self.tank.setProperties(data.get("tank", {}))
        self.feed.setProperties(data.get("feed", {}))
        self.injector.setProperties(data.get("injector", {}))
        self.combustion.setProperties(data.get("combustion", {}))
        self.nozzle.setProperties(data.get("nozzle", {}))
        self.config.setProperties(data.get("config", {}))
        self.grains = [self.grain]

    def getChamberVolume(self, regression):
        return (
            self.grain.getFreeVolume(regression)
            + self.config.getProperty("chamberDeadVolume")
        )

    def getFuelMass(self, regression):
        return (
            self.grain.getVolumeAtRegression(regression)
            * self.fuel.getProperty("density")
        )

    def getGeometryAlerts(self):
        alerts = []
        for alert in self.grain.getGeometryErrors():
            alert.location = "Fuel Grain"
            alerts.append(alert)
        alerts.extend(self.nozzle.getGeometryErrors())
        initialUllage = self.tank.getInitialUllageVolume(self.oxidizer)
        if initialUllage <= 0:
            alerts.append(
                SimAlert(
                    SimAlertLevel.ERROR,
                    SimAlertType.VALUE,
                    "Initial oxidizer volume must be smaller than tank volume",
                    "Oxidizer Tank",
                )
            )
        if self.tank.getProperty("residualMass") > self.tank.getProperty(
            "oxidizerMass"
        ):
            alerts.append(
                SimAlert(
                    SimAlertLevel.ERROR,
                    SimAlertType.VALUE,
                    "Residual oxidizer mass cannot exceed initial oxidizer mass",
                    "Oxidizer Tank",
                )
            )
        return alerts

    def _appendState(
        self,
        result,
        time,
        regression,
        oxidizerMass,
        fuelMass,
        chamberMass,
        chamberPressure,
        tankPressure,
        feedPressure,
        oxidizerFlow,
        fuelFlow,
        nozzleFlow,
        oxidizerFlux,
        force,
        exitPressure,
        throatChange,
    ):
        chamberVolume = self.getChamberVolume(regression)
        web = self.grain.getWebLeft(regression)
        portArea = self.grain.getPortArea(regression)
        totalFlow = oxidizerFlow + fuelFlow
        mixtureRatio = oxidizerFlow / fuelFlow if fuelFlow > 0 else 0
        totalInitialVolume = self.grain.getGrainBoundingVolume()
        volumeLoading = 100 * (
            1 - self.grain.getFreeVolume(regression) / totalInitialVolume
        )

        values = {
            "time": time,
            "kn": 0,
            "pressure": chamberPressure,
            "force": force,
            "mass": (fuelMass,),
            "volumeLoading": volumeLoading,
            "massFlow": (fuelFlow,),
            "massFlux": (oxidizerFlux,),
            "regression": (regression,),
            "web": (web,),
            "exitPressure": exitPressure,
            "dThroat": throatChange,
            "machNumber": (0,),
            "tankPressure": tankPressure,
            "feedPressure": feedPressure,
            "oxidizerMass": oxidizerMass,
            "oxidizerMassFlow": oxidizerFlow,
            "fuelMass": fuelMass,
            "fuelMassFlow": fuelFlow,
            "totalMassFlow": totalFlow,
            "nozzleMassFlow": nozzleFlow,
            "mixtureRatio": mixtureRatio,
            "oxidizerMassFlux": oxidizerFlux,
            "chamberGasMass": chamberMass,
            "chamberVolume": chamberVolume,
            "portArea": portArea,
        }
        for channel, value in values.items():
            result.channels[channel].addData(value)

    def runSimulation(self, callback=None):
        result = HybridSimulationResult(self)
        for alert in self.getGeometryAlerts():
            result.addAlert(alert)
        if result.getAlertsByLevel(SimAlertLevel.ERROR):
            return result

        self.grain.simulationSetup(self.config)
        dt = min(self.config.getProperty("timestep"), 0.001)
        ambient = self.config.getProperty("ambPressure")
        regression = 0
        oxidizerMass = self.tank.getProperty("oxidizerMass")
        fuelMass = self.getFuelMass(regression)
        chamberVolume = self.getChamberVolume(regression)
        initialPressure = max(
            self.combustion.getProperty("initialChamberPressure"), ambient
        )
        chamberMass = (
            initialPressure
            * chamberVolume
            / (
                self.combustion.getSpecificGasConstant()
                * self.combustion.getProperty("temperature")
            )
        )
        chamberPressure = initialPressure
        tankPressure = self.tank.getPressure(oxidizerMass, self.oxidizer)
        feedPressure = self.feed.getPressure(tankPressure)
        throatChange = 0
        self._appendState(
            result,
            0,
            regression,
            oxidizerMass,
            fuelMass,
            chamberMass,
            chamberPressure,
            tankPressure,
            feedPressure,
            0,
            0,
            0,
            0,
            0,
            0,
            throatChange,
        )

        dropoutReported = False
        reversePressureReported = False
        time = 0
        maxTime = self.config.getProperty("maxSimTime")
        residualMass = self.tank.getProperty("residualMass")
        initialOxidizerMass = oxidizerMass

        while time < maxTime:
            tankPressure = self.tank.getPressure(oxidizerMass, self.oxidizer)
            feedPressure = self.feed.getPressure(tankPressure)
            if (
                self.feed.getProperty("mode") == "Regulated"
                and tankPressure
                < self.feed.getProperty("regulatedPressure")
                + self.feed.getProperty("regulatorMargin")
                and not dropoutReported
            ):
                result.addAlert(
                    SimAlert(
                        SimAlertLevel.MESSAGE,
                        SimAlertType.VALUE,
                        "Regulator dropped out of regulation",
                        "Feed System",
                    )
                )
                dropoutReported = True

            oxidizerFlow = 0
            oxidizerFlux = 0
            fuelFlow = 0
            activeOxidizer = oxidizerMass > residualMass
            webLeft = self.grain.getWebLeft(regression)
            if activeOxidizer and webLeft > self.config.getProperty(
                "burnoutWebThres"
            ):
                oxidizerFlow = self.injector.getMassFlow(
                    feedPressure,
                    chamberPressure,
                    self.oxidizer.getProperty("density"),
                )
                availableOxidizer = max(oxidizerMass - residualMass, 0)
                oxidizerFlow = min(oxidizerFlow, availableOxidizer / dt)
                if feedPressure <= chamberPressure and not reversePressureReported:
                    result.addAlert(
                        SimAlert(
                            SimAlertLevel.WARNING,
                            SimAlertType.VALUE,
                            "Feed pressure did not exceed chamber pressure",
                            "Injector",
                        )
                    )
                    reversePressureReported = True

                portArea = self.grain.getPortArea(regression)
                if portArea > 0:
                    oxidizerFlux = oxidizerFlow / portArea
                regressionRate = self.fuel.getRegressionRate(oxidizerFlux)
                regressionStep = min(regressionRate * dt, max(webLeft, 0))
                nextFuelMass = self.getFuelMass(regression + regressionStep)
                fuelFlow = max((fuelMass - nextFuelMass) / dt, 0)
            else:
                regressionStep = 0
                nextFuelMass = fuelMass

            oxidizerMass -= oxidizerFlow * dt
            regression += regressionStep
            fuelMass = nextFuelMass
            chamberVolume = self.getChamberVolume(regression)
            throatArea = self.nozzle.getThroatArea(throatChange)
            gasTerm = (
                self.combustion.getSpecificGasConstant()
                * self.combustion.getProperty("temperature")
                / chamberVolume
            )
            outflowRateConstant = throatArea * gasTerm / max(
                self.combustion.getCStar(), 1e-12
            )
            inflow = oxidizerFlow + fuelFlow
            previousChamberMass = chamberMass
            if outflowRateConstant > 0:
                ambientMass = ambient / gasTerm
                steadyMass = ambientMass + inflow / outflowRateConstant
                chamberMass = steadyMass + (
                    chamberMass - steadyMass
                ) * math.exp(-outflowRateConstant * dt)
            else:
                chamberMass += inflow * dt
            nozzleFlow = max(
                (previousChamberMass + inflow * dt - chamberMass) / dt,
                0,
            )
            chamberPressure = (
                chamberMass * gasTerm
            )

            gamma = self.combustion.getProperty("gamma")
            if chamberPressure > ambient:
                exitPressure = self.nozzle.getExitPressure(gamma, chamberPressure)
                thrustCoefficient = self.nozzle.getAdjustedThrustCoeff(
                    chamberPressure,
                    ambient,
                    gamma,
                    throatChange,
                    exitPressure,
                )
                force = max(
                    thrustCoefficient * throatArea * chamberPressure,
                    0,
                )
            else:
                exitPressure = ambient
                force = 0

            if chamberPressure == 0:
                slagRate = 0
            else:
                slagRate = self.nozzle.getProperty("slagCoeff") / chamberPressure
            erosionRate = (
                chamberPressure * self.nozzle.getProperty("erosionCoeff")
            )
            throatChange += dt * ((-2 * slagRate) + (2 * erosionRate))
            time += dt

            self._appendState(
                result,
                time,
                regression,
                oxidizerMass,
                fuelMass,
                chamberMass,
                chamberPressure,
                tankPressure,
                feedPressure,
                oxidizerFlow,
                fuelFlow,
                nozzleFlow,
                oxidizerFlux,
                force,
                exitPressure,
                throatChange,
            )

            if callback is not None:
                consumedFraction = (
                    (initialOxidizerMass - oxidizerMass)
                    / max(initialOxidizerMass - residualMass, 1e-12)
                )
                if callback(min(max(consumedFraction, 0), 1)):
                    return result

            sourceActive = (
                oxidizerMass > residualMass
                and feedPressure > chamberPressure
                and self.grain.getWebLeft(regression)
                > self.config.getProperty("burnoutWebThres")
            )
            pressureDecayed = chamberPressure <= ambient * 1.001
            if not sourceActive and pressureDecayed:
                break

            if not math.isfinite(chamberPressure) or not math.isfinite(
                chamberMass
            ):
                result.addAlert(
                    SimAlert(
                        SimAlertLevel.ERROR,
                        SimAlertType.VALUE,
                        "Hybrid simulation produced a non-finite chamber state",
                        "Motor",
                    )
                )
                return result

        if time >= maxTime:
            result.addAlert(
                SimAlert(
                    SimAlertLevel.WARNING,
                    SimAlertType.CONSTRAINT,
                    "Simulation reached the configured maximum time",
                    "Motor",
                )
            )

        if result.getMaxPressure() > self.config.getProperty("maxPressure"):
            result.addAlert(
                SimAlert(
                    SimAlertLevel.WARNING,
                    SimAlertType.CONSTRAINT,
                    "Max pressure exceeded configured limit",
                    "Motor",
                )
            )
        if result.channels["oxidizerMassFlux"].getMax() > self.config.getProperty(
            "maxOxidizerFlux"
        ):
            result.addAlert(
                SimAlert(
                    SimAlertLevel.WARNING,
                    SimAlertType.CONSTRAINT,
                    "Peak oxidizer mass flux exceeded configured limit",
                    "Fuel Grain",
                )
            )

        result.success = not result.getAlertsByLevel(SimAlertLevel.ERROR)
        return result

    def getQuickResults(self):
        self.grain.simulationSetup(self.config)
        return {
            "volumeLoading": 100
            * (
                1
                - self.grain.getFreeVolume(0)
                / self.grain.getGrainBoundingVolume()
            ),
            "initialKn": 0,
            "propellantMass": self.getFuelMass(0)
            + self.tank.getProperty("oxidizerMass"),
            "portRatio": self.grain.getPortArea(0)
            / self.nozzle.getThroatArea(),
            "length": self.grain.getProperty("length"),
            "diameter": self.grain.getProperty("diameter"),
            "initialTankPressure": self.tank.getProperty("initialPressure"),
        }
