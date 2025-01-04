import ray
import subprocess
from pathlib import Path
from typing import Dict
from config import FACEFUSION_PATH

@ray.remote
class FaceFusionActor:
    def __init__(self):
        self.tasks: Dict[str, str] = {}

    async def process_face_fusion(
        self,
        task_id: str,
        source_path: str,
        target_path: str,
        output_path: str
    ) -> Dict[str, str]:
        try:
            command = [
                "python",
                str(FACEFUSION_PATH),
                "headless-run",
                "--processors",
                "face_swapper",
                "face_enhancer",
                "--face-selector-mode",
                "one",
                "--source-paths",
                source_path,
                "--target-path",
                target_path,
                "--output-path",
                output_path
            ]
            
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )
            
            if process.returncode == 0:
                self.tasks[task_id] = "completed"
                return {
                    "status": "completed",
                    "output_path": output_path
                }
        except subprocess.CalledProcessError as e:
            self.tasks[task_id] = "failed"
            return {
                "status": "failed",
                "error": str(e.stderr)
            }

    def get_task_status(self, task_id: str) -> str:
        return self.tasks.get(task_id, "not_found")