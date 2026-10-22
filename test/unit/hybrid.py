import math
import unittest

from motorlib.hybrid import FeedSystem, HybridMotor, Injector, Oxidizer, OxidizerTank
from motorlib.motor import Motor
from motorlib.motorFactory import motorFromDict


class HybridComponentMethods(unittest.TestCase):
    def test_tank_pressure_decreases_during_blowdown(self):
        oxidizer = Oxidizer({"density": 800})
        tank = OxidizerTank(
            {
                "volume": 0.01,
                "oxidizerMass": 4,
                "initialPressure": 5e6,
                "pressurantGamma": 1.2,
            }
        )

        self.assertAlmostEqual(tank.getPressure(4, oxidizer), 5e6)
        self.assertLess(tank.getPressure(2, oxidizer), 5e6)

    def test_regulated_feed_drops_to_upstream_pressure(self):
        feed = FeedSystem(
            {
                "mode": "Regulated",
                "regulatedPressure": 4e6,
                "regulatorMargin": 1e5,
                "dropoutBehavior": "Upstream Pressure",
            }
        )

        self.assertEqual(feed.getPressure(5e6), 4e6)
        self.assertEqual(feed.getPressure(3.5e6), 3.5e6)

    def test_injector_blocks_reverse_pressure(self):
        injector = Injector({"dischargeCoefficient": 0.7, "area": 1e-5})

        self.assertEqual(injector.getMassFlow(1e6, 2e6, 800), 0)
        self.assertGreater(injector.getMassFlow(2e6, 1e6, 800), 0)


class HybridMotorMethods(unittest.TestCase):
    def test_motor_factory_defaults_legacy_files_to_solid(self):
        solid = Motor()
        data = solid.getDict()
        del data["motorType"]

        self.assertIsInstance(motorFromDict(data), Motor)

    def test_hybrid_round_trip(self):
        motor = HybridMotor()
        motor.feed.setProperty("mode", "Regulated")
        restored = motorFromDict(motor.getDict())

        self.assertIsInstance(restored, HybridMotor)
        self.assertEqual(restored.feed.getProperty("mode"), "Regulated")
        self.assertEqual(restored.getDict(), motor.getDict())

    def test_simulation_conserves_total_mass(self):
        motor = HybridMotor()
        motor.config.setProperty("maxSimTime", 0.5)
        result = motor.runSimulation()

        initial = (
            result.channels["oxidizerMass"].getPoint(0)
            + result.channels["fuelMass"].getPoint(0)
            + result.channels["chamberGasMass"].getPoint(0)
        )
        final = (
            result.channels["oxidizerMass"].getLast()
            + result.channels["fuelMass"].getLast()
            + result.channels["chamberGasMass"].getLast()
        )
        exhausted = 0
        times = result.channels["time"].getData()
        flows = result.channels["nozzleMassFlow"].getData()
        for index in range(1, len(times)):
            exhausted += flows[index] * (times[index] - times[index - 1])

        self.assertTrue(result.success)
        self.assertAlmostEqual(initial, final + exhausted, places=6)

    def test_simulation_is_stable_above_internal_step_limit(self):
        impulses = []
        for timestep in (0.01, 0.005):
            motor = HybridMotor()
            motor.config.setProperty("timestep", timestep)
            motor.config.setProperty("maxSimTime", 0.5)
            impulses.append(motor.runSimulation().getImpulse())

        self.assertTrue(all(math.isfinite(value) for value in impulses))
        self.assertAlmostEqual(impulses[0], impulses[1], places=8)


if __name__ == "__main__":
    unittest.main()
