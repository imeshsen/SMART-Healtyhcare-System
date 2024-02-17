import React from "react";
import { useNavigate } from "react-router-dom";
import "./Navbar.css";

function Navbar() {
  const navigate = useNavigate();
  return (
    <div className="Navbar">
      <div className="name"><h1>SMART Healthcare System</h1></div>
      <div className="button">
        <button onClick={()=>{navigate("/")}}>Logout</button>
      </div>
    </div>
  );
}

export default Navbar;
