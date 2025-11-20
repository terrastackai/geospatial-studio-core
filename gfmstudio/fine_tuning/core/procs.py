# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import asyncio
from typing import Optional


class ProcessError(Exception): ...


async def check_output(*args: str, stdin: Optional[bytes] = None, **kwargs) -> str:
    """Function to check output

    Parameters
    ----------
    stdin : Optional[bytes], optional
        input, by default None

    Returns
    -------
    str
        decoded output

    Raises
    ------
    ProcessError


    Example
    -------
    >>> # stdin
    >>> print(await check_output("cat", "-n", stdin=b"one\ntwo\nthree\nnee\ntano\n"))
    >>> # no stdin
    >>> await check_output("ls", "-l")
    """
    if stdin is not None:
        kwargs["stdin"] = asyncio.subprocess.PIPE
    p = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **kwargs,
    )
    stdout, stderr = await p.communicate(input=stdin)
    if p.returncode == 0:
        return stdout.decode()
    else:
        stderr = stderr.decode()
        message = f"Error calling subprocess {args} {stderr}"
        raise ProcessError(message)
