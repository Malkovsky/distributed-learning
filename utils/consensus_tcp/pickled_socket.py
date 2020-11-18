import pickle, asyncio

class PickledSocketWrapper:
    SIZE_HEADER_LEN = 16
    BYTE_ORDER = 'little'

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.reader = reader
        self.writer = writer

    async def send(self, obj):
        data = pickle.dumps(obj)
        data_size = len(data).to_bytes(self.SIZE_HEADER_LEN, self.BYTE_ORDER)
        self.writer.write(data_size)
        await self.writer.drain()
        self.writer.write(data)
        await self.writer.drain()

    async def recv(self):
        data_size_bytes = await self.reader.readexactly(self.SIZE_HEADER_LEN)
        data_size = int.from_bytes(data_size_bytes, self.BYTE_ORDER)
        data = await self.reader.readexactly(data_size)
        return pickle.loads(data)
