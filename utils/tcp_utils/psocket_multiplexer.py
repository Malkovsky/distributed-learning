from typing import Dict, Any
import asyncio

from .pickled_socket import PickledSocketWrapper


class PSocketMultiplexer:
    def __init__(self, sockets: Dict[Any, PickledSocketWrapper]):
        self.sockets = sockets
        self._tasks: Dict[Any, asyncio.Task] = dict()

    async def _wrap_listen_with_token(self, token):
        resp = await self.sockets[token].recv()
        return (token, resp)

    def __aiter__(self):
        return self

    async def __anext__(self):  # returns (sender's token, msg, psocket)
        if len(self.sockets) == 0:
            raise StopAsyncIteration
        while True:
            for token in self.sockets.keys():
                if token not in self._tasks:
                    self._tasks[token] = asyncio.create_task(self._wrap_listen_with_token(token))
                else:
                    if self._tasks[token].done():
                        # todo: task's exception is re-raised! is it ok?
                        token, actual_result = self._tasks[token].result()
                        del self._tasks[token]
                        return token, actual_result, self.sockets[token]
            try:
                await asyncio.wait(list(self._tasks.values()), return_when=asyncio.FIRST_COMPLETED)
            except Exception as e:
                print("! PSOCKET_SELECTOR EXCEPTION: ", str(e))
                raise e
