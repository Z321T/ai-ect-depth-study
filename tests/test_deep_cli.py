from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class DeepCLITests(unittest.TestCase):
    def test_help_renders_percentages_from_another_working_directory(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/train_deep.py'
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, str(script), '--help'], cwd=tmp,
                                    text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('50% clean / 50% 10..30 dB', result.stdout)
        self.assertIn('--device {auto,cpu,cuda}', result.stdout)


if __name__ == '__main__':
    unittest.main()
