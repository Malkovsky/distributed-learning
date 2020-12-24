from typing import Dict, Any
import asyncio

from .pickled_socket import PickledSocketWrapper


class PSocketSelector:
    def __init__(self):
        self.sockets: Dict[Any, PickledSocketWrapper] = dict()
        self._tasks: Dict[Any, asyncio.Task] = dict()
        self._done = set()

    async def _wrap_listen_with_token(self, token):
        resp = await self.sockets[token].recv()
        return (token, resp)

    def add(self, token, psocket: PickledSocketWrapper):
        self.sockets[token] = psocket

    def remove(self, token):
        if token in self._tasks.keys():
            self._tasks[token].cancel()
            del self._tasks[token]

    async def recv(self, timeout=None):  # returns (token, actual result of recv)
        while True:
            if len(self.sockets) == 0:
                return (None, None)

            if len(self._done) > 0:
                token, actual_result = self._done.pop().result()
                if token in self._tasks.keys():
                    del self._tasks[token]
                return token, actual_result

            for token in self.sockets.keys():
                if token not in self._tasks:
                    self._tasks[token] = asyncio.create_task(self._wrap_listen_with_token(token))
            try:
                self._done, _ = await asyncio.wait(list(self._tasks.values()), return_when=asyncio.FIRST_COMPLETED, timeout=timeout)
            except TimeoutError:
                return (None, None)
