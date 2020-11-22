const WebSocketServer = require('ws').Server;
const wss = new WebSocketServer({port: 7000});

connections = []
waitingClient = []
var rooms={}
nextRoomID = 0


class connectedClient{
	constructor(ws, room){
		this.ws = ws
		this.roomID = room
	}

}

class Room{
	constructor(p1,p2, id){
		this.player1 = p1
		this.player2 = p2
		this.phase = 0
		this.id = id

		this.received1 = false
		this.received2 = false

		this.connecting1 = true
		this.connecting2 = true
	}
}

wss.on('connection', function(ws){
	
	myroomID = -1
	
	connections.push(new connectedClient(ws, -1))
	waitingClient.push(ws)
	console.log("%d players connecting", connections.length)

	var clientNum = connections.length
	//二人いたらマッチ成立
	if(waitingClient.length >= 2){
		//ルーム作成
		roomID = nextRoomID
		//覚えておけるか実験
		myroomID = nextRoomID
		nextRoomID += 1
		rooms[roomID] = new Room(waitingClient[0], waitingClient[1], roomID)
		
		//そいつらをconnectionsの待ち行列から消す
		waitingClient[0].send('you matched')
		waitingClient[0].send('roomID:'+roomID)
		waitingClient[1].send('you matched')
		waitingClient[1].send('roomID:'+roomID)

		//connectionsの中の2接続の所属ルームを決める
		for(let i in connections){
			if(connections[i].ws = waitingClient[0] || connections[i].ws == waitingClient[1]){
				connections[i].roomID = roomID
			}
		}
		waitingClient = connections.slice(2)
		console.log(Object.keys(rooms))
		//ws.send('hello')
	}

	ws.on('message', function(message){
		console.log('received: %s', message)
		//if(message.substring)
		message = JSON.parse(message)	
		//マッチングしてる部屋用の処理
		if("roomID" in message){
			if (!(message["roomID"] in rooms)){
				console.log("unknown room accessed")
				return 1
			}

			room = rooms[message["roomID"]]
			if(room.phase == 0 || room.phase == 2){
				//お題の単語交換フェーズ
				if(room.player1 == ws){
					room.received1 = true
					room.player2.send("word:"+message["content"])
				}
				else if(room.player2 == ws){
					room.received2 = true
					room.player1.send("word:"+message["content"])
				}
				else{
					console.log('unknown player send word')
				}
				if(room.received1 && room.received2){
					room.player1.send("make title")
					room.player2.send("make title")
					
					room.received1 = false
					room.received2 = false
					room.phase += 1
					console.log("room %d proceed to phase %d", room.id, room.phase)
				}
			}
			else if(room.phase==1 || room.phase == 3){
				//タイトル考えフェーズ
				if(room.player1 == ws){
					//viewCount = 100000
					//senderViewCount = 100000

					room.received1 = true
	                        	room.player2.send("rivalTitle:"+message["content"])
	                    		//room.player2.send("rivalViewCount:"+message["viewCount"])
					//room.player1.send("yourViewCount:"+senderViewCount)
				}
                                else if(room.player2 == ws){
					//viewCount = 100000
					//senderViewCount = 100000

					room.received2 = true
					room.player1.send("rivalTitle:"+message["content"])
					//room.player1.send("rivalViewCount:"+message["viewCount"])
					//room.player2.send("yourViewCount:"+senderViewCount)
				}
				else{   
					console.log('unknown player send title')
				}
				if(room.received1 && room.received2){
					room.player1.send("show result")
					room.player2.send("show result")
					room.phase += 1

					room.received1 = false
					room.received2 = false
					console.log("room %d proceed to phase %d", room.id, room.phase)
				}
			}
			if(room.phase >= 4){
				//試合終了
				room.player1.send("end game")
				room.player2.send("end game")
			}
		}
	})

	ws.on('close', function () {
		console.log('a connection was closed.')
		console.log("room %d was broken", myroomID)
		rival = null
		//roomに入っていれば相手に通知してroom削除
		if(myroomID != -1){
			room = rooms[myroomID]
			if(room.player1 == ws){
				room.player2.send("room broken")
				rival = room.player2
				room.connecting1 = false
			}
			else if(room.player2 == ws){
				room.player1.send("room broken")
				rival = room.player1
				room.connecting2 = false
			}
			if(!room.connecting1 && !room.connecting2){
				delete rooms[myroomID]
			}
		}
		//connection削除
		connections = connections.filter(function (conn, i) {
			return (conn.ws === ws || conn.ws == rival) ? false : true;
		});
		waitingClient = waitingClient.filter(function (conn, i) {
			return (conn === ws || conn.ws == rival) ? false : true;
		});
	});
});

