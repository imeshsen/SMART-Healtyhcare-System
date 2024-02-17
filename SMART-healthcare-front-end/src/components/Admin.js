import React, { useState } from "react";
import AdminNavbar from "./AdminNavbar";
import "./Admin.css";

export default function Admin() {
  const [type, setType] = useState("");
  const [name, setName] = useState("");
  const [specialist, setSpecialist] = useState("");

  const handleClick = (e) => {
    const data = {name, type , specialist}

    fetch("http://localhost:8080/api/v1/diseases/save",{
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
    }).then(()=>{
        alert("Success !");
    });
  };

  return (
    <div>
      <AdminNavbar />
      <div className="admin">
        <input
          type="text"
          value={name}
          placeholder="Disease name"
          onChange={(e) => setName(e.target.value)}
        ></input>
        <br /><br/>
        <input
          type="text"
          value={type}
          placeholder="Disease type"
          onChange={(e) => setType(e.target.value)}
          required
        ></input>
        <br /><br/>
        <input
          type="text"
          value={specialist}
          placeholder="Specialist"
          onChange={(e) => setSpecialist(e.target.value)}
        ></input>
        <br /><br/>
        <button onClick={handleClick}>Add</button>
      </div>
    </div>
  );
}
